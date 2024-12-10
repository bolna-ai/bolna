import time
import uuid

from .base_manager import BaseManager
from .task_manager import TaskManager
from bolna.helpers.logger_config import configure_logger
from bolna.models import AGENT_WELCOME_MESSAGE
from bolna.helpers.utils import update_prompt_with_context

logger = configure_logger(__name__)


class AssistantManager(BaseManager):
    def __init__(self, agent_config, ws=None, assistant_id=None, context_data=None, conversation_history=None,
                 turn_based_conversation=None, cache=None, input_queue=None, output_queue=None, **kwargs):
        super().__init__()
        self.run_id = str(uuid.uuid4())
        self.assistant_id = assistant_id
        self.tools = {}
        self.websocket = ws
        self.agent_config = agent_config
        self.context_data = context_data
        self.tasks = agent_config.get('tasks', [])
        self.task_states = [False] * len(self.tasks)
        self.turn_based_conversation = turn_based_conversation
        self.cache = cache
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.kwargs = kwargs
        self.conversation_history = conversation_history
        self.kwargs['agent_welcome_message'] = update_prompt_with_context(agent_config.get('agent_welcome_message', AGENT_WELCOME_MESSAGE), context_data)

    async def run(self, local=False, run_id=None):
        """
        Run will start all tasks in sequential format
        """
        if run_id:
            self.run_id = run_id

        input_parameters = None
        for task_id, task in enumerate(self.tasks):
            logger.info(f"Running task {task_id} {task} and sending kwargs {self.kwargs}")
            task_manager = TaskManager(self.agent_config.get("agent_name", self.agent_config.get("assistant_name")),
                                       task_id, task, self.websocket,
                                       context_data=self.context_data, input_parameters=input_parameters,
                                       assistant_id=self.assistant_id, run_id=self.run_id,
                                       turn_based_conversation=self.turn_based_conversation,
                                       cache=self.cache, input_queue=self.input_queue, output_queue=self.output_queue,
                                       conversation_history=self.conversation_history, **self.kwargs)
            await task_manager.load_prompt(self.agent_config.get("agent_name", self.agent_config.get("assistant_name")),
                                           task_id, local=local, **self.kwargs)
            task_output = await task_manager.run()
            task_output['run_id'] = self.run_id
            yield task_id, task_output.copy()
            self.task_states[task_id] = True
            if task_id == 0:
                input_parameters = task_output

                # removing context_data from non-conversational tasks
                self.context_data = None
            logger.info(f"task_output {task_output}")
            if task["task_type"] == "extraction":
                input_parameters["extraction_details"] = task_output["extracted_data"]
        logger.info("Done with execution of the agent")
