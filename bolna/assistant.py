from bolna.models import *
from bolna.agent_manager import AssistantManager


class Assistant:
    def __init__(self, name="trial_agent"):
        self.name = name
        self.tasks = []

    def add_task(
        self,
        task_type,
        llm_agent,
        input_queue=None,
        output_queue=None,
        transcriber=None,
        synthesizer=None,
        enable_textual_input=False,
    ):
        pipelines = []
        toolchain_args = {}
        tools_config_args = {}
        toolchain_args["execution"] = "parallel"
        toolchain_args["pipelines"] = pipelines
        tools_config_args["llm_agent"] = llm_agent
        tools_config_args["input"] = {"format": "wav", "provider": "default"}

        tools_config_args["output"] = {"format": "wav", "provider": "default"}
        if transcriber is None:
            pipelines.append(["llm"])
            tools_config_args["transcriber"] = transcriber

        pipeline = ["transcriber", "llm"]
        if synthesizer is not None:
            pipeline.append("synthesizer")
            tools_config_args["synthesizer"] = synthesizer
        pipelines.append(pipeline)

        if enable_textual_input:
            pipelines.append(["llm"])

        toolchain = ToolsChainModel(execution="parallel", pipelines=pipelines)
        task = Task(tools_config=ToolsConfig(**tools_config_args), toolchain=toolchain, task_type=task_type).dict()
        self.tasks.append(task)

    async def execute(self):
        agent_config = {"agent_name": self.name, "tasks": self.tasks}
        self.manager = AssistantManager(agent_config, ws=None, input_queue=None, output_queue=None)
        async for index, task_output in self.manager.run():
            yield task_output
