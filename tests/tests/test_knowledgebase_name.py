from bolna.agent_types.knowledgebase_agent import KnowledgeBaseAgent
from bolna.models import KnowledgeAgentConfig


def test_knowledge_agent_config_accepts_name_and_defaults_to_source():
    config = KnowledgeAgentConfig(
        rag_config={
            "vector_store": {"provider_config": {"vector_ids": ["kb-1", "kb-2"]}},
            "used_sources": [
                {"vector_id": "kb-1", "source": "https://docs.example.com", "name": "Docs"},
                {"vector_id": "kb-2", "source": "https://faq.example.com"},
            ],
        }
    )

    assert config.rag_config["used_sources"][0]["name"] == "Docs"
    assert config.rag_config["used_sources"][1]["name"] == "https://faq.example.com"


def test_initialize_rag_config_preserves_names_for_used_sources():
    agent = KnowledgeBaseAgent.__new__(KnowledgeBaseAgent)
    agent.config = {
        "rag_config": {
            "vector_store": {"provider_config": {"vector_ids": ["kb-1", "kb-2"]}},
            "used_sources": [
                {"vector_id": "kb-1", "source": "https://docs.example.com", "name": "Docs"},
                {"vector_id": "kb-2", "source": "https://faq.example.com"},
            ],
        }
    }

    rag_config = agent._initialize_rag_config()

    assert rag_config["collections"] == ["kb-1", "kb-2"]
    assert rag_config["used_sources"][0]["name"] == "Docs"
    assert rag_config["used_sources"][1]["name"] == "https://faq.example.com"


def test_knowledge_agent_config_preserves_unexpected_source_fields():
    config = KnowledgeAgentConfig(
        rag_config={
            "vector_store": {"provider_config": {"vector_ids": ["kb-1"]}},
            "used_sources": [
                {
                    "vector_id": "kb-1",
                    "source": "https://docs.example.com",
                    "name": "Docs",
                    "custom_metadata": {"category": "guides"},
                }
            ],
        }
    )

    assert config.rag_config["used_sources"][0]["custom_metadata"] == {"category": "guides"}
