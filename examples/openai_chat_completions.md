# Simple-agent streaming Chat Completions

Use this LLM fragment at `tasks[].tools_config.llm_agent.llm_config` with
`agent_type: simple_llm_agent`. It is not a complete voice-agent configuration.

```json
{
  "provider": "openai",
  "model": "YOUR_COMPATIBLE_REASONING_MODEL",
  "max_tokens": 4096,
  "reasoning_effort": "medium",
  "use_responses_api": false,
  "prompt_cache_key": "support-assistant-v1",
  "omit_request_parameters": ["temperature", "stop", "verbosity"]
}
```

Use a model supported by Bolna's existing reasoning-model mapping and your account.
Existing `max_tokens` mapping supplies its completion cap; no second cap setting
or new model detection is added. The existing service tier remains `default`.

The optional cache key is forwarded on streaming Chat requests. The omission
list removes only the three listed tuning fields after defaults are applied, never
messages, tools or the cap. Unset options preserve existing behavior. Explicit
`use_responses_api: false` overrides automatic transport selection for this simple
agent; absent/null retains it. Chat routing and non-streaming behavior are unchanged.
For Responses history, storage and strict WebSocket controls, see
[the Responses example](openai_responses_controls.md). Keep credentials in the existing environment.

Bolna forwards the caller-supplied `prompt_cache_key`; it does not automatically
distribute traffic across keys. Keep all turns of a call on the same key. If
high traffic reduces cache hits, distribute calls across a stable key pool using
a deterministic mapping, then tune the pool size using measured cache hits.
See [OpenAI's cache-key guidance](https://developers.openai.com/api/docs/guides/prompt-caching#prompt-cache-key-best-practices).

Tests use mocked SDK transport, not live model quality or voice validation.
[API fields](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
