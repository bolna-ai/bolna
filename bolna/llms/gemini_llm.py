import os
import json
import uuid
import base64
from typing import AsyncIterable
from google import genai
from google.genai import types
from bolna.constants import default_thinking_level
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import (
    now_ms,
    compute_function_pre_call_message,
    convert_to_request_log,
    clean_gemini_schema,
)
from .llm import BaseLLM
from .types import LLMStreamChunk, LatencyData, FunctionCallPayload

logger = configure_logger(__name__)


def _usage_kwargs(usage) -> dict:
    """Map Gemini usage_metadata onto the LLMStreamChunk token fields."""
    if not usage:
        return {}
    # Gemini keeps thinking tokens out of candidates_token_count; OpenAI folds them into
    # output_tokens, so add them here to keep billing consistent across providers.
    return {
        "input_tokens": usage.prompt_token_count,
        "output_tokens": (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
        "reasoning_tokens": usage.thoughts_token_count,
        "cached_tokens": usage.cached_content_token_count,
    }


class GeminiLLM(BaseLLM):
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
                            parameters=clean_gemini_schema(func["parameters"]),
                        )
                    )
                elif "name" in tool and "parameters" in tool:
                    gemini_declarations.append(
                        types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool.get("description", ""),
                            parameters=clean_gemini_schema(tool["parameters"]),
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
        """Thinking knob per family: 3.x takes thinking_level, 2.5 takes thinking_budget.

        Sending either one to the other family is a 400, so an explicit budget only
        applies to 2.5.
        """
        m = self.model

        if self.thinking_budget and self.thinking_budget > 0 and "2.5" in m:
            return types.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True)

        if m.startswith("gemini-3"):
            return types.ThinkingConfig(thinking_level=default_thinking_level(m), include_thoughts=True)

        if "2.5" in m:
            if "pro" in m:
                # Pro cannot disable thinking; 128 is its floor.
                return types.ThinkingConfig(thinking_budget=128, include_thoughts=True)
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
        self, messages, synthesize=True, meta_info=None, tool_choice=None, tools=None
    ) -> AsyncIterable[LLMStreamChunk]:
        # tools= accepted for interface parity; per-node scoping is not wired for Gemini.
        system_instruction, history = self._prepare_history(messages)
        config = self._build_config(system_instruction)

        start_time = now_ms()
        first_token_time = None
        latency_data = None  # set on first token, mirrors OpenAI pattern

        answer, buffer = "", ""
        self.started_streaming = False
        self.gave_out_prefunction_call_message = False
        accumulated_thought_parts: list[str] = []
        # Gemini 3 streams thought_signature as a standalone Part before the functionCall Part.
        pending_thought_signature: bytes | None = None
        # Keep last non-empty usage_metadata (some models omit it on the final chunk).
        stream_usage = None
        # Accumulate fn args per call_id across chunks; dispatch once post-stream with full args.
        _pending_fn_args: dict[str, dict] = {}
        _pending_dispatch: dict[str, dict] = {}
        _tool_dispatched = False

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

                # Read before the parts below, so a function_call yield carries this chunk's usage
                if chunk.usage_metadata:
                    stream_usage = chunk.usage_metadata

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
                            raw_id = part.function_call.id
                            call_id = raw_id or ("call_" + str(uuid.uuid4())[:8])
                            chunk_args = dict(part.function_call.args) if part.function_call.args else {}
                            if call_id not in _pending_fn_args:
                                _pending_fn_args[call_id] = {}
                            _pending_fn_args[call_id].update(chunk_args)
                            fn_args = _pending_fn_args[call_id]
                            # Rebuild Part with synthetic id to avoid id=None mismatch 400 on next turn.
                            if not raw_id:
                                inline_sig = getattr(part, "thought_signature", None)
                                rebuilt_kwargs: dict = dict(
                                    function_call=types.FunctionCall(name=fn_name, args=fn_args, id=call_id)
                                )
                                if inline_sig:
                                    rebuilt_kwargs["thought_signature"] = inline_sig
                                part = types.Part(**rebuilt_kwargs)
                            self._native_function_parts[call_id] = part
                            logger.info(
                                f"[GeminiLLM] function_call detected fn={fn_name} call_id={call_id} args={list(fn_args.keys())} native_part_cached=True"
                            )

                            # task_manager dispatches on the function-call chunk and abandons this
                            # generator, so the post-loop flush never runs.
                            if synthesize and buffer.strip():
                                yield LLMStreamChunk(data=buffer, end_of_stream=True, latency=latency_data)
                                buffer = ""

                            if not self.gave_out_prefunction_call_message:
                                pre_msg_config = self.api_params.get(fn_name, {}).get("pre_call_message")
                                detected_lang = meta_info.get("detected_language") if meta_info else None
                                active_lang = detected_lang or self.language
                                pre_msg = compute_function_pre_call_message(active_lang, fn_name, pre_msg_config)
                                self.gave_out_prefunction_call_message = True
                                if pre_msg:
                                    yield LLMStreamChunk(
                                        data=pre_msg,
                                        end_of_stream=True,
                                        latency=latency_data,
                                        function_name=fn_name,
                                        function_message=pre_msg_config,
                                    )

                            if call_id not in _pending_dispatch:
                                model_resp_prefix: list[dict] = [
                                    {"type": "_gemini_thought", "text": t} for t in accumulated_thought_parts
                                ]
                                accumulated_thought_parts = []
                                sig_bytes = getattr(part, "thought_signature", None) or pending_thought_signature
                                pending_thought_signature = None
                                _pending_dispatch[call_id] = {
                                    "fn_name": fn_name,
                                    "func_conf": self.api_params.get(fn_name, {}),
                                    "model_resp_prefix": model_resp_prefix,
                                    "sig_bytes": sig_bytes,
                                }
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

        if latency_data and latency_data.total_stream_duration_ms is None:
            latency_data.total_stream_duration_ms = now_ms() - start_time

        for call_id, ctx in _pending_dispatch.items():
            fn_name = ctx["fn_name"]
            fn_args = _pending_fn_args.get(call_id, {})
            func_conf = ctx["func_conf"]

            tool_spec = next(
                (
                    t
                    for t in self.bolna_tools_raw
                    if (t.get("type") == "function" and t["function"]["name"] == fn_name) or (t.get("name") == fn_name)
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
                    missing = [k for k in required_keys if k not in fn_args]
                    logger.warning(
                        f"[GeminiLLM] Tool call {fn_name} still missing params after full stream: "
                        f"missing={missing}, got={list(fn_args.keys())} — "
                        f"dispatching anyway (OpenAI-parity; downstream will validate)"
                    )

            model_resp: list[dict] = list(ctx["model_resp_prefix"])
            fn_entry: dict = {
                "id": call_id,
                "function": {"name": fn_name, "arguments": json.dumps(fn_args)},
                "type": "function",
            }
            sig_bytes = ctx["sig_bytes"]
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
            _tool_dispatched = True
            yield LLMStreamChunk(
                data=payload,
                end_of_stream=False,
                latency=latency_data,
                is_function_call=True,
                **_usage_kwargs(stream_usage),
            )

        reasoning_content = "\n".join(accumulated_thought_parts) if accumulated_thought_parts else None

        if synthesize and buffer.strip():
            yield LLMStreamChunk(
                data=buffer,
                end_of_stream=True,
                latency=latency_data,
                reasoning_content=reasoning_content,
                **_usage_kwargs(stream_usage),
            )
        elif synthesize and not buffer.strip() and not _tool_dispatched:
            logger.error(
                "[GeminiLLM] Dead turn detected: synthesize=True, buffer empty, no tool dispatched. "
                f"accumulated_args_keys={list(_pending_fn_args.keys())} answer={answer!r}"
            )
        elif not synthesize:
            yield LLMStreamChunk(
                data=answer,
                end_of_stream=True,
                latency=latency_data,
                reasoning_content=reasoning_content,
                **_usage_kwargs(stream_usage),
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
