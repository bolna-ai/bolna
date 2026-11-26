"""The knowledge base is queried with what the caller said, not with whatever came last.

After a tool call, the follow-up generation runs on a history that ends in the tool result,
so querying RAG with history[-1] searched the knowledge base for the tool's JSON instead of
the caller's question, and the reply after every tool call lost its knowledge-base context.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from bolna.agent_types.knowledgebase_agent import KnowledgeBaseAgent
from bolna.helpers.utils import latest_user_message

sys.path.insert(0, str(Path(__file__).parent))
from test_expression_routing import _make_agent  # noqa: E402

QUESTION = "Is Dr Rao free on Friday, and how much is a cleaning?"
TOOL_RESULT = '{"status": "success", "slots": ["10:00", "15:30"]}'

HISTORY_AFTER_TOOL_CALL = [
    {"role": "user", "content": QUESTION},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "check_slots", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "c1", "content": TOOL_RESULT},
]


class _FakeRAGClient:
    def __init__(self):
        self.queries = []

    async def query_for_conversation(self, query, **kwargs):
        self.queries.append(query)
        response = MagicMock()
        response.contexts = []
        response.total_query_time_ms = 1
        response.server_processing_time_ms = 1
        response.total_results = 0
        return response


def _patch_rag(module, client):
    return patch(f"{module}.RAGServiceClientSingleton.get_client", AsyncMock(return_value=client))


def test_latest_user_message_skips_tool_and_assistant_turns():
    assert latest_user_message(HISTORY_AFTER_TOOL_CALL) == QUESTION


def test_latest_user_message_returns_the_last_user_turn():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    assert latest_user_message(history) == "second"


def test_latest_user_message_without_a_user_turn_is_empty():
    assert latest_user_message([]) == ""
    assert latest_user_message(None) == ""
    assert latest_user_message([{"role": "assistant", "content": "hello"}]) == ""


def test_graph_agent_queries_rag_with_the_user_question_after_a_tool_call():
    agent = _make_agent()
    agent.global_rag_config = {"collections": ["kb1"]}
    agent.rag_configs = {}
    client = _FakeRAGClient()

    with _patch_rag("bolna.agent_types.graph_agent", client):
        asyncio.run(agent._build_messages(HISTORY_AFTER_TOOL_CALL))

    assert client.queries == [QUESTION]


def test_knowledgebase_agent_queries_rag_with_the_user_question_after_a_tool_call():
    agent = KnowledgeBaseAgent.__new__(KnowledgeBaseAgent)
    agent.rag_config = {"collections": ["kb1"]}
    agent.rag_server_url = "http://rag.invalid"
    client = _FakeRAGClient()

    with _patch_rag("bolna.agent_types.knowledgebase_agent", client):
        asyncio.run(agent._add_rag_context(HISTORY_AFTER_TOOL_CALL))

    assert client.queries == [QUESTION]
