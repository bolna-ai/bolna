import random
import json
import asyncio
import traceback
from .base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import update_prompt_with_context, get_md5_hash

logger = configure_logger(__name__)


class Node:
    def __init__(self, node_id, node_label, content, classification_labels: list = None, prompt=None, milestone_check_prompt=None,
                 children: list = None):
        self.node_id = node_id
        self.node_label = node_label
        self.content = content
        self.children = children
        self.classification_labels = classification_labels
        self.prompt = prompt
        self.need_response_generation = False
        self.milestone_check_prompt = milestone_check_prompt


class Graph:
    def __init__(self, conversation_data, preprocessed=False, context_data=None):
        self.preprocessed = preprocessed
        self.root = None
        self.graph = self._create_graph(conversation_data, context_data)

    def _create_graph(self, data, context_data=None):
        logger.info("Creating graph")
        node_map = dict()
        for node_id, node_data in data.items():
            prompt_parts = node_data.get("prompt").split('###Examples')
            prompt = node_data.get('prompt')
            if len(prompt_parts) == 2:
                classification_prompt = prompt_parts[0]
                user_prompt = update_prompt_with_context(prompt_parts[1], context_data)
                prompt = '###Examples'.join([classification_prompt, user_prompt])

            node = Node(
                node_id=node_id,
                node_label=node_data["label"],
                content=node_data["content"],
                classification_labels=node_data.get("classification_labels", []),
                prompt=prompt,
                children=[],
                milestone_check_prompt=node_data.get("milestone_check_prompt", ""),
            )
            node_map[node_id] = node
            if node_data["is_root"]:
                self.root = node

        for node_id, node_data in data.items():
            children_ids = node_data.get("children", [])
            node_map[node_id].children = [node_map[child_id] for child_id in children_ids] if children_ids else []
        return node_map

    # @TODO complete this function
    def remove_node(self, parent, node):
        print("Not yet implemented")


class GraphBasedConversationAgent(BaseAgent):
    def __init__(self, llm, prompts, context_data=None, preprocessed=True):
        super().__init__()
        # Config
        self.llm = llm
        self.context_data = context_data
        self.preprocessed = preprocessed

        # Goals
        self.graph = None
        self.conversation_intro_done = False

    def load_prompts_and_create_graph(self, prompts):
        self.graph = Graph(prompts,  context_data=self.context_data)
        self.current_node = self.graph.root
        self.current_node_interim = self.graph.root #Handle interim node because we are dealing with interim results 

    def _get_audio_text_pair(self, node):
        ind = random.randint(0, len(node.content) - 1)
        audio_pair = node.content[ind]
        contextual_text = update_prompt_with_context(audio_pair['text'], self.context_data)
        if contextual_text != audio_pair['text']:
            audio_pair['text'] = contextual_text
            audio_pair['audio'] = get_md5_hash(contextual_text)
        return audio_pair

    async def _get_next_formulaic_agent_next_step(self, history, stream=True, synthesize=False):
        pass

    def _handle_intro_message(self):
        audio_pair = self._get_audio_text_pair(self.current_node)
        self.conversation_intro_done = True
        logger.info("Conversation intro done")
        if self.current_node.prompt is None:
            #These are convos with two step intros
            ind = random.randint(0, len(self.current_node.children) - 1)
            self.current_node = self.current_node.children[ind]
        return audio_pair

    async def _get_next_preprocessed_step(self, history):
        logger.info(f"current node {self.current_node.node_label}, self.current_node == self.graph.root {self.current_node == self.graph.root}, and self.conversation_intro_done {self.conversation_intro_done}")
        if self.current_node == self.graph.root and not self.conversation_intro_done:
            return self._handle_intro_message()

        logger.info("Conversation intro was done and hence moving forward")
        if len(history) > 7:
            # @TODO: Add summary of left out messages
            prev_messages = history[-6:]
        else:
            prev_messages = history[1:]

        message = [{"role": "system", "content": self.current_node.prompt}] + prev_messages
        # Get classification label from LLM
        response = await self.llm.generate(message, request_json=True)
        logger.info(f"Classification response {response}")
        classification_result = json.loads(response)
        label = classification_result["classification_label"]
        for child in self.current_node.children:
            if child.node_label.strip().lower() == label.strip().lower():
                self.current_node_interim = child
                return self._get_audio_text_pair(child)

    def update_current_node(self):
        self.current_node = self.current_node_interim
    
    # Label flow is not being used right now as we're logging every request
    async def generate(self, history, label_flow=None):
        try:
            if self.preprocessed:
                logger.info(f"Current node {str(self.current_node)}")
                if len(self.current_node.children) == 0:
                    ind = random.randint(0, len(self.current_node.content) - 1)
                    audio_pair = self.current_node.content[ind]
                    logger.info('Agent: {}'.format(audio_pair.get('text')))
                    yield audio_pair
                else:
                    next_state = await self._get_next_preprocessed_step(history)
                    logger.info('Agent: {}'.format(next_state))
                    yield next_state
                
                if len(self.current_node.children) == 0:
                    await asyncio.sleep(1)
                    yield "<end_of_conversation>"
            else:
                # @TODO add non-preprocessed flow
                async for message in self._get_next_formulaic_agent_next_step(history, True, False):
                    yield message

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error sending intro text: {e}")
