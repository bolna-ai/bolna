import os
import json
import uuid
import base64
from typing import AsyncIterable
from google import genai
from google.genai import types
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import now_ms, compute_function_pre_call_message, convert_to_request_log
from .llm import BaseLLM
from .types import LLMStreamChunk, LatencyData, FunctionCallPayload

logger = configure_logger(__name__)


class GeminiLLM(BaseLLM):
    def _clean_schema(self, schema):
        """Gemini Protos don't support additionalProperties."""
        if not isinstance(schema, dict):
            return schema
        cleaned = {k: v for k, v in schema.items() if k != "additionalProperties"}
        for k, v in cleaned.items():
            if isinstance(v, dict):
                cleaned[k] = self._clean_schema(v)
            elif isinstance(v, list):
                cleaned[k] = [self._clean_schema(i) if isinstance(i, dict) else i for i in v]
        return cleaned

    def __init__(self, max_tokens=100, buffer_size=40, model="gemini-2.5-flash", temperature=0.1, **kwargs):
        super().__init__(max_tokens, buffer_size)

        # New SDK uses plain model names like "gemini-2.0-flash", no "models/" prefix
        self.model = model
        if "/" in model:
            self.model = model.split("/")[-1]
        if self.model.startswith("models/"):
            self.model = self.model[len("models/") :]

        self.temperature = temperature
        api_key = kwargs.get("llm_key", os.getenv("GOOGLE_API_KEY"))
        self.client = genai.Client(api_key=api_key)

        self.api_params = kwargs.get("api_tools", {}).get("tools_params", {})
        bolna_tools = kwargs.get("api_tools", {}).get("tools", [])

        gemini_declarations = []
        if bolna_tools:
            if isinstance(bolna_tools, str):
                try:
                    bolna_tools = json.loads(bolna_tools)
                except json.JSONDecodeError:
                    logger.error("Failed to parse tool definitions as JSON")
                    bolna_tools = []

            for tool in bolna_tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    gemini_declarations.append(
                        types.FunctionDeclaration(
                            name=func["name"],
                            description=func["description"],
                            parameters=self._clean_schema(func["parameters"]),
                        )
                    )
                elif "name" in tool and "parameters" in tool:
                    gemini_declarations.append(
                        types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool.get("description", ""),
                            parameters=self._clean_schema(tool["parameters"]),
                        )
                    )

        self.gemini_tools = [types.Tool(function_declarations=gemini_declarations)] if gemini_declarations else None
        # Keep raw bolna tools list for required-param validation at call time
        self.bolna_tools_raw = bolna_tools if isinstance(bolna_tools, list) else []
        self.thinking_budget = kwargs.get("thinking_budget", 0)
        self.run_id = kwargs.get("run_id", None)
        self.language = kwargs.get("language", "en")
        # Cache of original types.Part objects keyed by function call id.
        # Gemini 3 thought_signatures cannot survive bytes serialisation — the only
        # reliable way to return them is to reuse the exact Part object the SDK gave us.
        self._native_function_parts: dict[str, types.Part] = {}
        logger.info(
            f"[GeminiLLM] Initialized model={self.model} tools={[d.name for d in gemini_declarations] if gemini_declarations else None} thinking_budget={self.thinking_budget}"
        )

    def _prepare_history(self, messages):
        """Translate Bolna roles (OpenAI-style) to Gemini-style roles and parts."""
        system_instruction = ""
        history = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            if role == "system":
                system_instruction = content
                continue

            parts = []
            if content:
                parts.append(types.Part(text=content))

            if tool_calls:
                for tc in tool_calls:
                    if tc.get("type") == "_gemini_thought":
                        parts.append(types.Part(thought=True, text=tc.get("text", "")))
                    elif tc.get("type") == "function":
                        fn = tc["function"]
                        call_id = tc.get("id")
                        # If we have the original SDK Part object cached, reuse it directly.
                        # This is the only reliable way to preserve thought_signature for
                        # Gemini 3 models — any byte-level reconstruction corrupts the signature.
                        native = self._native_function_parts.get(call_id) if call_id else None
                        if native is not None:
                            # Best path: reuse the exact SDK Part object — no serialisation risk.
                            parts.append(native)
                            logger.info(
                                f"[GeminiLLM] _prepare_history: native Part cache HIT for call_id={call_id} fn={fn['name']}"
                            )
                        else:
                            # Fallback (e.g. after server restart when cache is empty):
                            # reconstruct the Part from stored JSON fields.
                            # SDK >=1.68.0 encodes thought_signature bytes as standard base64
                            # so this round-trip is now safe.
                            args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                            fc_kwargs = dict(name=fn["name"], args=args)
                            if call_id:
                                fc_kwargs["id"] = call_id
                            part_kwargs: dict = dict(function_call=types.FunctionCall(**fc_kwargs))
                            has_sig = bool(tc.get("thought_signature"))
                            if has_sig:
                                part_kwargs["thought_signature"] = base64.b64decode(tc["thought_signature"])
                            parts.append(types.Part(**part_kwargs))
                            logger.info(
                                f"[GeminiLLM] _prepare_history: native Part cache MISS for call_id={call_id} fn={fn['name']} reconstructed=True thought_signature={has_sig}"
                            )

            if role == "assistant":
                if parts:
                    history.append(types.Content(role="model", parts=parts))
            elif role == "user":
                if parts:
                    history.append(types.Content(role="user", parts=parts))
            elif role == "tool":
                tool_name = msg.get("name")
                if not tool_name:
                    # Recover tool name from the preceding model turn's function_call part
                    if history and history[-1].role == "model":
                        for p in history[-1].parts:
                            if p.function_call:
                                tool_name = p.function_call.name
                                break

                if tool_name:
                    try:
                        resp_obj = json.loads(content) if isinstance(content, str) else content
                        if not isinstance(resp_obj, dict):
                            resp_obj = {"result": content}
                    except Exception:
                        resp_obj = {"result": content}

                    fr_kwargs = dict(name=tool_name, response=resp_obj)
                    # Docs require matching the exact id from the function_call
                    tool_call_id = msg.get("tool_call_id")
                    if tool_call_id:
                        fr_kwargs["id"] = tool_call_id
                    history.append(
                        types.Content(
                            role="user", parts=[types.Part(function_response=types.FunctionResponse(**fr_kwargs))]
                        )
                    )
                else:
                    history.append(types.Content(role="user", parts=[types.Part(text=f"Tool result: {content}")]))

        return system_instruction, history

    def _get_thinking_config(self) -> "types.ThinkingConfig | None":
        """
        Return the correct ThinkingConfig per model family based on Google docs:

        Gemini 3.x  → use thinking_level (NOT thinking_budget)
          - Pro    : level="low"     (minimum; "minimal" not supported for Pro)
          - Flash/Lite: level="minimal" (near-off; avoids thought_signature overhead)

        Gemini 2.5  → use thinking_budget
          - Pro    : budget=128      (minimum; cannot disable)
          - Flash  : budget=0        (fully off; range 0-24576)
          - Flash-Lite: budget=0     (fully off; range 512-24576 but 0 disables)

        If the user explicitly passed a thinking_budget > 0, honour it (2.5 models).
        """
        m = self.model

        # User explicitly set a budget — honour it for 2.5 models only.
        # Gemini 3.x uses thinking_level, not thinking_budget; passing budget to 3.x causes 400s.
        if self.thinking_budget and self.thinking_budget > 0 and "2.5" in m:
            return types.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True)

        # --- Gemini 3.x family: use thinking_level ---
        if m.startswith("gemini-3"):
            if "pro" in m:
                # Pro cannot disable thinking; "minimal" is not supported — use "low"
                return types.ThinkingConfig(thinking_level="low", include_thoughts=True)
            else:
                # Flash / Flash-Lite: "minimal" is closest to off, avoids thought_signature cost
                return types.ThinkingConfig(thinking_level="minimal", include_thoughts=True)

        # --- Gemini 2.5 family: use thinking_budget ---
        if "2.5" in m:
            if "pro" in m:
                # Pro cannot disable thinking; minimum budget is 128
                return types.ThinkingConfig(thinking_budget=128, include_thoughts=True)
            else:
                # Flash / Flash-Lite: disable with 0
                return types.ThinkingConfig(thinking_budget=0)

        return None

    def _build_config(self, system_instruction, request_json=False):
        config_kwargs = dict(
            system_instruction=system_instruction or None,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            response_mime_type="application/json" if request_json else "text/plain",
        )

        thinking_config = self._get_thinking_config()
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config

        config = types.GenerateContentConfig(**config_kwargs)
        if self.gemini_tools:
            config.tools = self.gemini_tools
            config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
        return config

    async def generate_stream(
        self, messages, synthesize=True, meta_info=None, tool_choice=None
    ) -> AsyncIterable[LLMStreamChunk]:
        system_instruction, history = self._prepare_history(messages)
        config = self._build_config(system_instruction)

        start_time = now_ms()
        first_token_time = None
        latency_data = None  # set on first token, mirrors OpenAI pattern

        answer, buffer = "", ""
        self.started_streaming = False
        self.gave_out_prefunction_call_message = False
        # Accumulated thought text parts from the current model turn (thinking models only)
        accumulated_thought_parts: list[str] = []
        # Standalone thought_signature bytes received before the function_call Part
        # (Gemini 3 streaming sends signature as a separate Part ahead of functionCall)
        pending_thought_signature: bytes | None = None

        try:
            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=history,
                config=config,
            )
            async for chunk in response_stream:
                now = now_ms()
                if not first_token_time:
                    first_token_time = now
                    self.started_streaming = True
                    latency_data = LatencyData(
                        sequence_id=meta_info.get("sequence_id") if meta_info else None,
                        first_token_latency_ms=first_token_time - start_time,
                    )

                # Check for function calls, thought parts, and signature parts
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        # Collect thought parts (thinking models show reasoning before function calls)
                        if getattr(part, "thought", False):
                            thought_text = getattr(part, "text", None)
                            if thought_text:
                                accumulated_thought_parts.append(thought_text)
                            continue

                        # Gemini 3 streaming: thought_signature arrives as a standalone Part
                        # (separate from the functionCall Part). Buffer it so we can attach
                        # it to the next function_call entry — the API requires it on the
                        # same functionCall Part, not as a separate history Part.
                        standalone_sig = getattr(part, "thought_signature", None)
                        if standalone_sig and not part.function_call:
                            pending_thought_signature = standalone_sig
                            continue

                        if part.function_call:
                            fn_name = part.function_call.name
                            fn_args = dict(part.function_call.args) if part.function_call.args else {}
                            raw_id = part.function_call.id
                            call_id = raw_id or ("call_" + str(uuid.uuid4())[:8])
                            # If Gemini didn't return a function_call id, rebuild the Part
                            # with our generated id — otherwise the cached Part would have
                            # id=None while the tool response carries the synthetic call_id,
                            # which causes an id-mismatch 400 on the next turn.
                            if not raw_id:
                                inline_sig = getattr(part, "thought_signature", None)
                                rebuilt_kwargs: dict = dict(
                                    function_call=types.FunctionCall(name=fn_name, args=fn_args, id=call_id)
                                )
                                if inline_sig:
                                    rebuilt_kwargs["thought_signature"] = inline_sig
                                part = types.Part(**rebuilt_kwargs)
                            # Cache the Part so _prepare_history can reuse it intact —
                            # thought_signature bytes cannot survive serialisation.
                            self._native_function_parts[call_id] = part
                            logger.info(
                                f"[GeminiLLM] function_call detected fn={fn_name} call_id={call_id} args={list(fn_args.keys())} native_part_cached=True"
                            )

                            if not self.gave_out_prefunction_call_message:
                                pre_msg_config = self.api_params.get(fn_name, {}).get("pre_call_message")
                                active_lang = (
                                    meta_info.get("detected_language", self.language) if meta_info else self.language
                                )
                                pre_msg = compute_function_pre_call_message(active_lang, fn_name, pre_msg_config)
                                yield LLMStreamChunk(
                                    data=pre_msg,
                                    end_of_stream=True,
                                    latency=latency_data,
                                    function_name=fn_name,
                                    function_message=pre_msg_config,
                                )
                                self.gave_out_prefunction_call_message = True

                            func_conf = self.api_params.get(fn_name, {})

                            tool_spec = next(
                                (
                                    t
                                    for t in self.bolna_tools_raw
                                    if (t.get("type") == "function" and t["function"]["name"] == fn_name)
                                    or (t.get("name") == fn_name)
                                ),
                                None,
                            )
                            if tool_spec:
                                params_schema = (
                                    tool_spec["function"]["parameters"]
                                    if tool_spec.get("type") == "function"
                                    else tool_spec.get("parameters", {})
                                )
                                required_keys = params_schema.get("required", [])
                                if not all(k in fn_args for k in required_keys):
                                    logger.warning(
                                        f"Gemini tool call {fn_name} missing required params: {required_keys}, got: {list(fn_args.keys())}"
                                    )
                                    continue

                            model_resp: list[dict] = []
                            for thought_text in accumulated_thought_parts:
                                model_resp.append({"type": "_gemini_thought", "text": thought_text})
                            accumulated_thought_parts = []

                            fn_entry: dict = {
                                "id": call_id,
                                "function": {"name": fn_name, "arguments": json.dumps(fn_args)},
                                "type": "function",
                            }
                            # Per Google docs: thought_signature must be returned on the
                            # same functionCall Part. In streaming Gemini 3 emits it as a
                            # standalone Part just before the functionCall Part; Gemini 2.5
                            # (when thinking on) puts it inline on the functionCall Part.
                            sig_bytes = getattr(part, "thought_signature", None) or pending_thought_signature
                            pending_thought_signature = None  # consumed
                            if sig_bytes:
                                fn_entry["thought_signature"] = base64.b64encode(sig_bytes).decode("utf-8")
                                logger.info(
                                    f"[GeminiLLM] thought_signature stored for fn={fn_name} call_id={call_id} bytes={len(sig_bytes)}"
                                )
                            model_resp.append(fn_entry)

                            payload = FunctionCallPayload(
                                url=func_conf.get("url"),
                                method=(func_conf.get("method", "GET") or "GET").lower(),
                                param=func_conf.get("param"),
                                api_token=func_conf.get("api_token"),
                                headers=func_conf.get("headers"),
                                model_args={"model": self.model},
                                meta_info=meta_info or {},
                                called_fun=fn_name,
                                model_response=model_resp,
                                tool_call_id=call_id,
                                textual_response=answer.strip() if answer else None,
                            )
                            for k, v in fn_args.items():
                                setattr(payload, k, v)

                            convert_to_request_log(
                                json.dumps(fn_args),
                                meta_info,
                                self.model,
                                "llm",
                                direction="response",
                                is_cached=False,
                                run_id=self.run_id,
                            )
                            yield LLMStreamChunk(
                                data=payload, end_of_stream=False, latency=latency_data, is_function_call=True
                            )
                            continue

                # Regular text streaming
                text_chunk = ""
                try:
                    text_chunk = chunk.text
                except (ValueError, IndexError, AttributeError):
                    pass

                if text_chunk:
                    answer += text_chunk
                    buffer += text_chunk

                    if synthesize and len(buffer) >= self.buffer_size:
                        split = buffer.rsplit(" ", 1)
                        yield LLMStreamChunk(data=split[0], end_of_stream=False, latency=latency_data)
                        buffer = split[1] if len(split) > 1 else ""

        except Exception as e:
            logger.error(f"Gemini unexpected error: {e}")
            raise
        finally:
            self.started_streaming = False

        if latency_data:
            latency_data.total_stream_duration_ms = now_ms() - start_time

        reasoning_content = "\n".join(accumulated_thought_parts) if accumulated_thought_parts else None

        if synthesize and buffer.strip():
            yield LLMStreamChunk(
                data=buffer, end_of_stream=True, latency=latency_data, reasoning_content=reasoning_content
            )
        elif not synthesize:
            yield LLMStreamChunk(
                data=answer, end_of_stream=True, latency=latency_data, reasoning_content=reasoning_content
            )

    async def generate(self, messages, request_json=False, ret_metadata=False):
        """Non-streaming — used for voicemail detection and completion checks."""
        system_instruction, history = self._prepare_history(messages)
        config = self._build_config(system_instruction, request_json=request_json)

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=history,
            config=config,
        )
        res = response.text
        return (res, {}) if ret_metadata else res
