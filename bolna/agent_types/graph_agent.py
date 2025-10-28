import os
import time
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import json

from bolna.models import *
from bolna.agent_types.base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.rag_service_client import RAGServiceClient, RAGServiceClientSingleton
from bolna.helpers.utils import now_ms

from typing import List, Tuple, Generator, AsyncGenerator

load_dotenv()
logger = configure_logger(__name__)

class GraphAgent(BaseAgent):
    def __init__(self, config: GraphAgentConfig):
        super().__init__()
        self.config = config
        self.agent_information = self.config.get('agent_information')
        self.current_node_id = self.config.get('current_node_id')
        self.context_data = self.config.get('context_data', {})
        self.llm_model = self.config.get('model')
        self.openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.node_history = ["root"]
        self.node_structure = self.build_node_structure()
        self.rag_configs = self.initialize_rag_configs()
        self.rag_server_url = os.getenv('RAG_SERVER_URL', 'http://localhost:8000')

    def initialize_rag_configs(self) -> Dict[str, Dict]:
        """Initialize RAG configurations for each node."""
        rag_configs = {}
        for node in self.config.get('nodes', []):
            rag_config = node.get('rag_config')
            if rag_config:
                # Extract collection/vector IDs
                collections = []
                # Legacy: direct vector_id on rag_config
                if 'vector_id' in rag_config:
                    collections.append(rag_config['vector_id'])
                # Legacy: rag_config.provider_config.vector_id
                elif 'provider_config' in rag_config and isinstance(rag_config.get('provider_config'), dict) and 'vector_id' in rag_config['provider_config']:
                    collections.append(rag_config['provider_config']['vector_id'])
                # New: rag_config.vector_store.provider_config.vector_id
                else:
                    try:
                        vector_store = rag_config.get('vector_store') or {}
                        provider_config = vector_store.get('provider_config') or {}
                        vs_vector_id = provider_config.get('vector_id')
                        if vs_vector_id:
                            collections.append(vs_vector_id)
                    except Exception:
                        pass
                
                rag_configs[node['id']] = {
                    'collections': collections,
                    'similarity_top_k': rag_config.get('similarity_top_k', 10),
                    'temperature': rag_config.get('temperature', 0.7),
                    'model': rag_config.get('model', 'gpt-4o'),
                    'max_tokens': rag_config.get('max_tokens', 150)
                }
                
                logger.info(f"Initialized RAG config for node {node['id']} with collections: {collections}")
                
        return rag_configs

    def build_node_structure(self) -> Dict[str, List[str]]:
        structure = {}
        for node in self.config.get('nodes', []):
            structure[node['id']] = [edge['to_node_id'] for edge in node.get('edges', [])]
        return structure

    def get_accessible_nodes(self, current_node_id: str) -> List[str]:
        accessible_nodes = []
        for node_id, children in self.node_structure.items():
            if current_node_id in children or node_id == current_node_id:
                logger.info(f"Node Id : {node_id} is accessible")
                accessible_nodes.extend([node_id] + children)
        return list(set(accessible_nodes))

    def get_node_by_id(self, node_id: str) -> Optional[dict]:
        return next((node for node in self.config.get('nodes', []) if node['id'] == node_id), None)

    async def generate_response(self, history: List[dict]) -> dict:
        current_node = self.get_node_by_id(self.current_node_id)
        if not current_node:
            raise ValueError("Current node is not found in the configuration.")

        prompt = current_node['prompt']
        rag_config = self.rag_configs.get(self.current_node_id)

        if rag_config and rag_config.get('collections'):
            try:
                client = await RAGServiceClientSingleton.get_client(self.rag_server_url)
                latest_message = history[-1]["content"]

                rag_response = await client.query_for_conversation(
                    query=latest_message,
                    collections=rag_config['collections'],
                    max_results=rag_config.get('similarity_top_k', 10),
                    similarity_threshold=0.0
                )

                if not rag_response.contexts or len(rag_response.contexts) == 0:
                    return await self._generate_standard_response(prompt, history)

                top_score = rag_response.contexts[0].score if rag_response.contexts else 0.0
                logger.info(f"RAG (node {self.current_node_id}): Retrieved {rag_response.total_results} contexts, top score: {top_score:.3f}")

                rag_context = await client.format_context_for_prompt(rag_response.contexts)
                return await self._generate_response_with_knowledge(prompt, history, rag_context)

            except asyncio.TimeoutError:
                logger.error(f"RAG service timeout for node {self.current_node_id}")
                return await self._generate_standard_response(prompt, history)

            except Exception as e:
                logger.error(f"RAG service error: {e}")
                return await self._generate_standard_response(prompt, history)

        else:
            return await self._generate_standard_response(prompt, history)


    async def _generate_standard_response(self, node_prompt: str, history: List[dict]) -> dict:
        """Generate response using node prompt + conversation history without RAG."""
        max_history = 50
        history_subset = history[-max_history:] if len(history) > max_history else history

        messages = [{"role": "system", "content": node_prompt}] + [
            {"role": item["role"], "content": item["content"]} for item in history_subset
        ]

        response = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=self.config.get('max_tokens', 150),
            temperature=self.config.get('temperature', 0.7),
            top_p=self.config.get('top_p', 1.0),
            frequency_penalty=self.config.get('frequency_penalty', 0),
            presence_penalty=self.config.get('presence_penalty', 0),
        )
        response_text = response.choices[0].message.content
        return {"role": "assistant", "content": response_text}

    async def _generate_response_with_knowledge(self, node_prompt: str, history: List[dict], rag_context: str) -> dict:
        """Generate response with node prompt augmented by RAG knowledge."""
        augmented_prompt = f"""{node_prompt}

You have access to relevant information from the knowledge base:

{rag_context}

Use this information naturally when it helps answer the user's questions. Don't force references if not relevant to the conversation."""

        max_history = 50
        history_subset = history[-max_history:] if len(history) > max_history else history

        messages = [{"role": "system", "content": augmented_prompt}] + [
            {"role": item["role"], "content": item["content"]} for item in history_subset
        ]

        response = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=self.config.get('max_tokens', 150),
            temperature=self.config.get('temperature', 0.7),
            top_p=self.config.get('top_p', 1.0),
            frequency_penalty=self.config.get('frequency_penalty', 0),
            presence_penalty=self.config.get('presence_penalty', 0),
        )
        response_text = response.choices[0].message.content
        return {"role": "assistant", "content": response_text}

    async def _generate_fallback_response(self, prompt: str, history: List[dict]) -> dict:
        """
        Deprecated: Use _generate_standard_response instead.
        Kept for backward compatibility.
        """
        latest_message = history[-1]["content"]
        system_message = f"{prompt}\n\nPlease respond based on the latest user message: '{latest_message}'."

        messages = [{"role": "system", "content": system_message}] + [
            {"role": item["role"], "content": item["content"]} for item in history[-5:]
        ]
        response = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=self.config.get('max_tokens', 150),
            temperature=self.config.get('temperature', 0.7),
            top_p=self.config.get('top_p', 1.0),
            frequency_penalty=self.config.get('frequency_penalty', 0),
            presence_penalty=self.config.get('presence_penalty', 0),
        )
        response_text = response.choices[0].message.content
        return {"role": "assistant", "content": response_text}
    
    def is_response_valid(self, response: str) -> bool:
        if not response or len(response.strip()) < 5:  
            return False
        generic_responses = [
            "I don't know", "I'm not sure", "Can you rephrase?",
            "Sorry, I didn't understand", "I'm unable to assist"
        ]
        if any(generic in response.lower() for generic in generic_responses):
            logger.info("Response contains a generic fallback phrase.")
            return False

    async def decide_next_move_cyclic(self, history: List[dict]) -> Optional[str]:
        current_node = self.get_node_by_id(self.current_node_id)
        accessible_nodes = self.get_accessible_nodes(current_node['id'])

        node_info = {}
        for node_id in accessible_nodes:
            node = self.get_node_by_id(node_id)
            if node:
                node_info[node_id] = {
                    "prompt": node['prompt'],
                }
        
        prompt = f"""
        Analyze the conversation in a {self.agent_information} and determine the user's intent based on the conversation history and their latest message.

        Latest Message from user : {history[-1]["content"]}
        Current node: {current_node['id']}
        Accessible nodes and their information: {json.dumps(node_info, indent=2)}

        Respond with the ID of the accessible nodes that best matches the user's intent, or "current" if the current node is still appropriate.
        For example, if the user's intent is "x" node then write "x" as the output. 

        NOTE: Don't write anything other than node id. No strings and sentences.
        """

        messages = [{"role": "system", "content": prompt}] + [{"role": item["role"], "content": item["content"]} for item in history[-3:]]

        try:
            response = self.openai.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=self.config.get('max_tokens', 150),
                temperature=self.config.get('temperature', 0.7),
            )
        except Exception as e:
            print(f"Error generating response: {e}")

        next_node_id = response.choices[0].message.content.strip().lower()
        logger.info(f"Next Node is : {next_node_id}")

        if next_node_id == "current" or next_node_id not in accessible_nodes:
            logger.info(f"No conditions met")
            return None
        
        return next_node_id

    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator[Tuple[str, bool, Optional[Dict], bool, None, None], None]:
        meta_info = kwargs.get('meta_info')
        start_time = now_ms()
        first_token_time = None
        buffer = ""
        buffer_size = 20
        latency_data = {
            "sequence_id": meta_info.get("sequence_id") if meta_info else None,
            "first_token_latency_ms": None,
            "total_stream_duration_ms": None
        }

        try:
            # Decide next move
            next_node_id = await self.decide_next_move_cyclic(message)
            if next_node_id:
                self.current_node_id = next_node_id
                if self.current_node_id == "end":
                    response_text = "\nThank you for using our service. Goodbye!"

            response = await self.generate_response(message)
            response_text = response["content"]

            words = response_text.split()
            for i, word in enumerate(words):
                if first_token_time is None:
                    first_token_time = now_ms()
                    latency_data["first_token_latency_ms"] = first_token_time - start_time

                buffer += word + " "

                if len(buffer.split()) >= buffer_size or i == len(words) - 1:
                    is_final = (i == len(words) - 1)
                    if is_final and latency_data:
                        latency_data["total_stream_duration_ms"] = now_ms() - start_time
                    yield buffer.strip(), is_final, latency_data, False, None, None
                    buffer = ""

            if buffer:
                if latency_data:
                    latency_data["total_stream_duration_ms"] = now_ms() - start_time
                yield buffer.strip(), True, latency_data, False, None, None

        except Exception as e:
            logger.error(f"Error in generate function: {e}")
            latency_data["first_token_latency_ms"] = latency_data.get("first_token_latency_ms") or 0
            latency_data["total_stream_duration_ms"] = now_ms() - start_time
            yield f"An error occurred: {str(e)}", True, latency_data, False, None, None
