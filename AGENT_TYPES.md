## Agent types supported

This project supports multiple LLM agent types you can configure in `tools_config.llm_agent.agent_type`.

- **simple_llm_agent**: Streaming contextual conversational agent for direct prompt → response. Implemented and used by default.
- **knowledgebase_agent**: Retrieval-Augmented Generation (RAG) over a vector store for grounded answers. Implemented.
- **graph_agent**: Node/edge guided conversation flow with optional per-node RAG. Implemented.
- **llm_agent_graph**: Orchestrates LLM tools via a graph-style controller. Defined in models; not wired into runtime yet.
- **multiagent**: Coordinates multiple sub-agents. Defined in models; not wired into runtime yet.

Notes
- Current runtime wiring for agent dispatch happens in `bolna/agent_manager/task_manager.py` and supports: `simple_llm_agent`, `knowledgebase_agent`, `graph_agent`.
- Validation of acceptable `agent_type` strings is defined in `bolna/models.py` (`LlmAgent.validate_llm_config`).
- Internally, there are additional agent classes (e.g., `StreamingContextualAgent`, `RAGAgent`, `GraphAgent`, `SummarizationContextualAgent`, `ExtractionContextualAgent`, `WebhookAgent`, `GraphBasedConversationAgent`), but only the five strings above are valid values for `tools_config.llm_agent.agent_type`.

Example
```json
"llm_agent": {
  "agent_type": "simple_llm_agent",
  "agent_flow_type": "streaming",
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "max_tokens": 200
  }
}
```


