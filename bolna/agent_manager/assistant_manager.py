import asyncio
import copy
import time
import uuid

from .base_manager import BaseManager
from .task_manager import TaskManager
from bolna.helpers.logger_config import configure_logger
from bolna.integrations import PostCallContext, run_post_call_integrations
from bolna.models import AGENT_WELCOME_MESSAGE, IntegrationConfig
from bolna.helpers.utils import update_prompt_with_context

logger = configure_logger(__name__)


class AssistantManager(BaseManager):
    def __init__(
        self,
        agent_config,
        ws=None,
        assistant_id=None,
        context_data=None,
        conversation_history=None,
        turn_based_conversation=None,
        cache=None,
        input_queue=None,
        output_queue=None,
        **kwargs,
    ):
        super().__init__()
        self.run_id = str(uuid.uuid4())
        self.assistant_id = assistant_id
        self.tools = {}
        self.websocket = ws
        self.agent_config = agent_config
        self.context_data = context_data
        self.tasks = agent_config.get("tasks", [])
        self.task_states = [False] * len(self.tasks)
        self.turn_based_conversation = turn_based_conversation
        self.cache = cache
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.kwargs = kwargs
        self.conversation_history = conversation_history
        # keep strong refs so fire-and-forget integration tasks are not GC'd
        self._background_tasks: set = set()
        if kwargs.get("is_web_based_call", False):
            self.kwargs["agent_welcome_message"] = agent_config.get("agent_welcome_message", AGENT_WELCOME_MESSAGE)
        else:
            self.kwargs["agent_welcome_message"] = update_prompt_with_context(
                agent_config.get("agent_welcome_message", AGENT_WELCOME_MESSAGE), context_data
            )

    async def run(self, local=False, run_id=None):
        """
        Run will start all tasks in sequential format
        """
        if run_id:
            self.run_id = run_id

        input_parameters = None
        all_task_outputs = []
        for task_id, task in enumerate(self.tasks):
            logger.info(f"Running task {task_id}")
            task_manager = TaskManager(
                self.agent_config.get("agent_name", self.agent_config.get("assistant_name")),
                task_id,
                task,
                self.websocket,
                context_data=self.context_data,
                input_parameters=input_parameters,
                assistant_id=self.assistant_id,
                run_id=self.run_id,
                turn_based_conversation=self.turn_based_conversation,
                cache=self.cache,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                conversation_history=self.conversation_history,
                **self.kwargs,
            )
            await task_manager.load_prompt(
                self.agent_config.get("agent_name", self.agent_config.get("assistant_name")),
                task_id,
                local=local,
                **self.kwargs,
            )
            task_output = await task_manager.run()
            task_output["run_id"] = self.run_id
            all_task_outputs.append(task_output)
            yield task_id, copy.deepcopy(task_output)
            self.task_states[task_id] = True
            if task_id == 0:
                input_parameters = task_output
            if task["task_type"] == "extraction":
                input_parameters["extraction_details"] = task_output["extracted_data"]

        try:
            self._fire_post_call_integrations(all_task_outputs)
        except Exception as e:
            logger.error(f"failed to schedule post-call integrations: {e}")

        logger.info("Done with execution of the agent")

    def _collect_integration_configs(self):
        seen = set()
        configs = []
        for task in self.tasks:
            tools = task.get("tools_config") or {}
            for raw in (tools.get("integrations") or []):
                cfg = raw if isinstance(raw, IntegrationConfig) else IntegrationConfig(**raw)
                if cfg.provider in seen:
                    logger.warning(
                        f"duplicate integration provider '{cfg.provider}' discarded; keeping first config"
                    )
                    continue
                seen.add(cfg.provider)
                configs.append(cfg)
        return configs

    def _build_post_call_context(self, all_task_outputs):
        primary = all_task_outputs[0] if all_task_outputs else {}
        ctx = PostCallContext(
            agent_name=self.agent_config.get("agent_name", self.agent_config.get("assistant_name", "")) or "",
            run_id=self.run_id,
            call_sid=primary.get("call_sid"),
            duration_seconds=primary.get("conversation_time"),
            hangup_reason=str(primary["hangup_detail"]) if primary.get("hangup_detail") else None,
            recording_url=primary.get("recording_url"),
        )
        for output in all_task_outputs:
            if output.get("task_type") == "summarization" and output.get("summary"):
                ctx.summary = output["summary"]
            elif output.get("task_type") == "extraction" and output.get("extracted_data"):
                ctx.extracted_data = output["extracted_data"]
        return ctx

    def _fire_post_call_integrations(self, all_task_outputs):
        configs = self._collect_integration_configs()
        if not configs:
            return
        ctx = self._build_post_call_context(all_task_outputs)
        task = asyncio.create_task(run_post_call_integrations(configs, ctx))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
