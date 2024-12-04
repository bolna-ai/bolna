import os
import time
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

from bolna.agent_types.knowledgebase_agent import RAGAgent
from bolna.models import *
from bolna.agent_types.base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger

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
        self.rag_agents = self.initialize_rag_agents()

    def initialize_rag_agents(self) -> Dict[str, RAGAgent]:
        rag_agents = {}
        for node in self.config.get('nodes', []):
            rag_config = node.get('rag_config')
            if rag_config:
                rag_agents[node['id']] = RAGAgent(
                    provider_config=rag_config,
                    temperature=rag_config.get('temperature', 0.7),
                    model=rag_config.get('model', 'gpt-4'),
                    max_tokens=rag_config.get('max_tokens', 150)
                )
        return rag_agents

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
        rag_agent = self.rag_agents.get(self.current_node_id)

        if rag_agent:
            logger.info(f"Using RAGAgent for node {self.current_node_id}")

            rag_context = ""

            try:
                async for chunk, is_final, latency, truncated in rag_agent.generate(history):
                    if is_final:
                        rag_context = chunk
                        break

                if not rag_context:
                    logger.warning(f"No relevant context retrieved from RAG for node {self.current_node_id}")
                    return await self._generate_fallback_response(prompt, history)

                # Emphasize the latest message
                latest_message = history[-1]["content"]
                combined_context = f"{prompt}\n\nHere is some additional information:\n{rag_context}\n\nPlease respond based on the latest user message: '{latest_message}'."
                logger.info(f"Combined context: {combined_context}")

                return await self._generate_response_from_openai(combined_context, history)

            except asyncio.TimeoutError:
                logger.error(f"Timeout occurred while fetching data from RAG for node {self.current_node_id}")
                return await self._generate_fallback_response(prompt, history)

            except Exception as e:
                logger.error(f"An error occurred while processing the RAG agent: {e}")
                return await self._generate_fallback_response(prompt, history)

        else:
            logger.info(f"No RAGAgent for node {self.current_node_id}, using prompt directly.")
            return await self._generate_fallback_response(prompt, history)


    async def _generate_response_from_openai(self, context: str, history: List[dict]) -> dict:
        """
        Generate response using OpenAI API with the provided context and explicitly focus on the latest user message.
        """
        latest_message = history[-1]["content"]
        system_message = f"{context}\n\nPlease respond based on the latest user message: '{latest_message}'."
        
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


    async def _generate_fallback_response(self, prompt: str, history: List[dict]) -> dict:
        """
        Generate a fallback response that emphasizes the latest user message.
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
        logger.info(f"Current node: {current_node}")
        accessible_nodes = self.get_accessible_nodes(current_node['id'])
        logger.info(f"Accessible nodes: {accessible_nodes}")

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
        logger.info(f"Next node logic message: {messages}")

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

    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator[Tuple[str, bool, float, bool], None]:
        logger.info(f"Generating response for message: {message}")
        start_time = time.time()
        first_token_time = None
        buffer = ""
        buffer_size = 20  # Default buffer size of 20 words
        try:
            # Generate response
            
            
            # Decide next move
            next_node_id = await self.decide_next_move_cyclic(message)
            if next_node_id:
                logger.info(f"Next node: {next_node_id}")
                self.current_node_id = next_node_id
                if self.current_node_id == "end":
                    response_text = "\nThank you for using our service. Goodbye!"
                    
            response = await self.generate_response(message)
            response_text = response["content"]

            words = response_text.split()
            for i, word in enumerate(words):
                if first_token_time is None:
                    first_token_time = time.time()
                    latency = first_token_time - start_time
                
                buffer += word + " "
                
                if len(buffer.split()) >= buffer_size or i == len(words) - 1:
                    is_final = (i == len(words) - 1)
                    yield buffer.strip(), is_final, latency, False
                    buffer = ""
            
            if buffer:
                yield buffer.strip(), True, latency, False

        except Exception as e:
            logger.error(f"Error in generate function: {e}")
            yield f"An error occurred: {str(e)}", True, time.time() - start_time, False