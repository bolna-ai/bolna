# Explicit Responses request and transport controls

For a simple agent, put this fragment in `tools_config.llm_agent`. Select a model
that supports your requested reasoning effort and Responses configuration:

```json
{
  "agent_type": "simple_llm_agent",
  "agent_flow_type": "streaming",
  "llm_config": {
    "model": "YOUR_COMPATIBLE_REASONING_MODEL",
    "provider": "openai",
    "max_tokens": 4096,
    "reasoning_effort": "medium",
    "use_responses_api": true,
    "prompt_cache_key": "example-agent-v1",
    "omit_request_parameters": [
      "temperature",
      "stop",
      "verbosity"
    ],
    "responses_store": false,
    "responses_history": "full",
    "responses_omit_parameters": [
      "truncation",
      "include"
    ],
    "strict_websocket": true,
    "compact_threshold": null
  }
}
```

`max_tokens` already maps to `max_output_tokens`; the provider's default service
tier is `default`. Full history uses the existing Chat-message adapter for system, user, assistant
and tool messages, with `previous_response_id: null` and text output format. It does
not replay hidden response/reasoning items. Supply the complete supported-role history each turn; arbitrary Responses input
items and developer-role messages are not supported by this adapter.
JSON output and explicitly configured tools remain intact. Compaction stays off
when its threshold is unset. `stop` is not emitted by the Responses builder;
`verbosity` omission removes only `text.verbosity`, not `text.format`.

Storage and history are independent. Without these opt-ins, existing storage,
chaining, truncation, reasoning include and fallback defaults remain unchanged.
[OpenAI's WebSocket guide](https://developers.openai.com/api/docs/guides/websocket-mode)
describes `store: false` and full-context starts with a null previous response ID.

## Strict text evaluation

Use `generate_stream(..., synthesize=False, meta_info=metadata)` on the native
OpenAI endpoint, with a fresh metadata dict per turn. Strict mode rejects HTTP
and non-streaming `generate`, disables automatic response replay and HTTP fallback,
and raises on failed/incomplete responses or a stream without completion. It does
not undo text already yielded before a failure. Existing connection reuse,
reconnection, cancellation and tool handling are retained; strict mode does not
promise successful server cancellation or disable connection establishment retries.
Always close the provider in `finally`.

`metadata["responses_ws_attempts"]` retains each strict attempt's transport, status,
response ID, reported model/tier, and raw terminal `usage`, including unknown fields.
Missing usage (`null`) is not zero usage or an empty object. Record failed attempts
as well as successful ones when accounting for cost. `first_visible_text_ms` starts
at the first nonempty text delta, not tool arguments or an empty delta. It and
`duration_ms` are measured from provider request start, including setup that occurs
inside the request but excluding eager connection work already completed. They are
not handshake-only, ASR, TTS or telephony timings. Existing chunk latency fields
retain their original first-event semantics.

The example is a configuration contract, not a live quality or voice acceptance
result. The no-network tests verify request projection and transport behavior.
