import os
import asyncio
import uuid
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query,Body
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from dotenv import load_dotenv
from bolna.helpers.utils import store_file
from bolna.prompts import *
from bolna.helpers.logger_config import configure_logger
from bolna.models import *
from bolna.llms import LiteLLM
from bolna.agent_manager.assistant_manager import AssistantManager

load_dotenv()
logger = configure_logger(__name__)

redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)
active_websockets: List[WebSocket] = []

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class CreateAgentPayload(BaseModel):
    agent_config: AgentModel
    agent_prompts: Optional[Dict[str, Dict[str, str]]]


@app.get("/agent/{agent_id}")
async def get_agent(agent_id: str):
    """Fetches an agent's information by ID."""
    try:
        agent_data = await redis_client.get(agent_id)
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent not found")

        return json.loads(agent_data)

    except Exception as e:
        logger.error(f"Error fetching agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")



@app.post("/agent")
async def create_agent(agent_data: CreateAgentPayload):
    agent_uuid = str(uuid.uuid4())
    data_for_db = agent_data.agent_config.model_dump()
    data_for_db["assistant_status"] = "seeding"
    agent_prompts = agent_data.agent_prompts
    logger.info(f'Data for DB {data_for_db}')

    if len(data_for_db['tasks']) > 0:
        logger.info("Setting up follow up tasks")
        for index, task in enumerate(data_for_db['tasks']):
            if task['task_type'] == "extraction":
                extraction_prompt_llm = os.getenv("EXTRACTION_PROMPT_GENERATION_MODEL")
                extraction_prompt_generation_llm = LiteLLM(model=extraction_prompt_llm, max_tokens=2000)
                extraction_prompt = await extraction_prompt_generation_llm.generate(
                    messages=[
                        {'role': 'system', 'content': EXTRACTION_PROMPT_GENERATION_PROMPT},
                        {'role': 'user', 'content': data_for_db["tasks"][index]['tools_config']["llm_agent"]['extraction_details']}
                    ])
                data_for_db["tasks"][index]["tools_config"]["llm_agent"]['extraction_json'] = extraction_prompt

    stored_prompt_file_path = f"{agent_uuid}/conversation_details.json"
    await asyncio.gather(
        redis_client.set(agent_uuid, json.dumps(data_for_db)),
        store_file(file_key=stored_prompt_file_path, file_data=agent_prompts, local=True)
    )

    return {"agent_id": agent_uuid, "state": "created"}


@app.put("/agent/{agent_id}")
async def edit_agent(agent_id: str, agent_data: CreateAgentPayload = Body(...)):
    """Edits an existing agent based on the provided agent_id."""
    try:

        existing_data = await redis_client.get(agent_id)
        if not existing_data:
            raise HTTPException(status_code=404, detail="Agent not found")

        existing_data = json.loads(existing_data)


        new_data = agent_data.agent_config.model_dump()
        new_data["assistant_status"] = "updated"
        agent_prompts = agent_data.agent_prompts

        logger.info(f"Updating Agent {agent_id}: {new_data}")


        for index, task in enumerate(new_data.get("tasks", [])):
            if task.get("task_type") == "extraction":
                extraction_prompt_llm = os.getenv("EXTRACTION_PROMPT_GENERATION_MODEL")
                if not extraction_prompt_llm:
                    raise HTTPException(status_code=500, detail="Extraction model not configured")

                extraction_prompt_generation_llm = LiteLLM(model=extraction_prompt_llm, max_tokens=2000)
                extraction_details = task["tools_config"]["llm_agent"].get("extraction_details", "")

                extraction_prompt = await extraction_prompt_generation_llm.generate(
                    messages=[
                        {"role": "system", "content": EXTRACTION_PROMPT_GENERATION_PROMPT},
                        {"role": "user", "content": extraction_details}
                    ]
                )

                new_data["tasks"][index]["tools_config"]["llm_agent"]["extraction_json"] = extraction_prompt


        stored_prompt_file_path = f"{agent_id}/conversation_details.json"
        await asyncio.gather(
            redis_client.set(agent_id, json.dumps(new_data)),
            store_file(file_key=stored_prompt_file_path, file_data=agent_prompts, local=True)
        )

        return {"agent_id": agent_id, "state": "updated"}

    except Exception as e:
        logger.error(f"Error updating agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/agent/{agent_id}")
async def delete_agent(agent_id: str):
    """Deletes an agent by ID."""
    try:
        agent_exists = await redis_client.exists(agent_id)
        if not agent_exists:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        await redis_client.delete(agent_id)
        return {"agent_id": agent_id, "state": "deleted"}

    except Exception as e:
        logger.error(f"Error deleting agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/all")
async def get_all_agents():
    """Fetches all agents stored in Redis."""
    try:

        agent_keys = await redis_client.keys("*")  
        
        if not agent_keys:
            return {"agents": []}  
        agents_data = []
        for key in agent_keys:
            try:
                data = await redis_client.get(key)
                agents_data.append(data)
            except Exception as e:
                logger.error(f"An error occurred with key {key}: {e}")


        agents = [{ "agent_id": key, "data": json.loads(data) } for key, data in zip(agent_keys, agents_data) if data]

        return {"agents": agents}

    except Exception as e:
        logger.error(f"Error fetching all agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


############################################################################################# 
# Websocket 
#############################################################################################
@app.websocket("/chat/v1/{agent_id}")
async def websocket_endpoint(agent_id: str, websocket: WebSocket, user_agent: str = Query(None)):
    logger.info("Connected to ws")
    await websocket.accept()
    active_websockets.append(websocket)
    agent_config, context_data = None, None
    try:
        retrieved_agent_config = await redis_client.get(agent_id)
        logger.info(
            f"Retrieved agent config: {retrieved_agent_config}")
        agent_config = json.loads(retrieved_agent_config)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=404, detail="Agent not found")

    assistant_manager = AssistantManager(agent_config, websocket, agent_id)

    try:
        async for index, task_output in assistant_manager.run(local=True):
            logger.info(task_output)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"error in executing {e}")
