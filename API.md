# Bolna API Documentation

## Endpoints

### Get Agent
Retrieves an agent's information by agent id.

**Endpoint:** `GET /agent/{agent_id}`

**Parameters:**
- `agent_id` (path) - string, required: Unique identifier of the agent

### Create Agent
Creates a new agent with specified configuration.

**Endpoint:** `POST /agent`

**Request Body:**
```json
{
  "agent_config": {
    "agent_name": "Alfred",
    "agent_type": "other",
    "tasks": [
      {
        "task_type": "conversation",
        "toolchain": {
          "execution": "parallel",
          "pipelines": [["transcriber", "llm", "synthesizer"]]
        },
        "tools_config": {
          "input": { "format": "wav", "provider": "twilio" },
          "output": { "format": "wav", "provider": "twilio" },
          "transcriber": {
            "encoding": "linear16",
            "language": "en",
            "provider": "deepgram",
            "stream": true
          },
          "llm_agent": {
            "agent_type": "simple_llm_agent",
            "agent_flow_type": "streaming",
            "llm_config": {
              "provider": "openai",
              "model": "gpt-4o-mini",
              "request_json": true
            }
          },
          "synthesizer": {
            "audio_format": "wav",
            "provider": "elevenlabs",
            "stream": true,
            "provider_config": {
              "voice": "George",
              "model": "eleven_turbo_v2_5",
              "voice_id": "JBFqnCBsd6RMkjVDRZzb"
            },
            "buffer_size": 100.0
          }
        },
        "task_config": {
          "hangup_after_silence": 30.0
        }
      }
    ],
    "agent_welcome_message": "How are you doing Bruce?"
  },
  "agent_prompts": {
    "task_1": {
      "system_prompt": "Why Do We Fall, Sir? So That We Can Learn To Pick Ourselves Up."
    }
  }
}
```

**Response:**
200 OK
```json
{
  "agent_id": "uuid-string",
  "state": "created"
}
```

### Edit Agent
Updates an existing agent's configuration.

**Endpoint:** `PUT /agent/{agent_id}`

**Parameters:**
- `agent_id` (path) - string, required: Unique identifier of the agent

**Request Body:**
Same as Create Agent endpoint


### Delete Agent
Deletes an agent from the system.

**Endpoint:** `DELETE /agent/{agent_id}`

**Parameters:**
- `agent_id` (path) - string, required: Unique identifier of the agent

**Response:**
200 OK
```json
{
  "agent_id": "string",
  "state": "deleted"
}
```


### Create Batch
Creates a new batch for an agent to initiate calls to a list of recipients.

**Endpoint:** `POST /batches`

**Headers:**
- `Authorization` - Bearer token: Your API key

**Request Body (multipart/form-data):**
- `agent_id` (string, required): Unique identifier of the agent to use for the batch calls
- `file` (file, required): CSV file containing recipient details
- `from_phone_numbers` (list of strings, required): List of phone numbers to use as caller IDs. Pass multiple values to use multiple originating numbers.

**Example:**

```bash
curl --location 'https://api.bolna.ai/batches' \
--header 'Authorization: Bearer <api_key>' \
--form 'agent_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"' \
--form 'file=@"/path/to/file"' \
--form 'from_phone_numbers="+919876543210"' \
--form 'from_phone_numbers="+919876543211"'
```

**Python Example:**

```python
import os
import asyncio
import aiohttp

host = "https://api.bolna.ai"
api_key = "<api_key>"
agent_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
file_path = "/path/to/file.csv"
schedule_time = "2024-06-01T04:10:00+05:30"
from_phone_numbers = ["+919876543210", "+919876543211"]


async def schedule_batch(api_key, batch_id, scheduled_at):
    url = f"{host}/batches/{batch_id}/schedule"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"scheduled_at": scheduled_at}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            return await response.json()


async def get_batch_status(api_key, batch_id):
    url = f"{host}/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            return await response.json()


async def get_batch_executions(api_key, batch_id):
    url = f"{host}/batches/{batch_id}/executions"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            return await response.json()


async def create_batch():
    url = f"{host}/batches"
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(file_path, "rb") as f:
        form_data = aiohttp.FormData()
        form_data.add_field("agent_id", agent_id)
        form_data.add_field("file", f, filename=os.path.basename(file_path))

        # Add multiple from_phone_numbers
        for phone in from_phone_numbers:
            form_data.add_field("from_phone_numbers", phone)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=form_data) as response:
                response_data = await response.json()
                if response_data.get("state") == "created":
                    batch_id = response_data.get("batch_id")
                    res = await schedule_batch(api_key, batch_id, scheduled_at=schedule_time)
                    if res.get("state") == "scheduled":
                        check = True
                        while check:
                            await asyncio.sleep(60)
                            res = await get_batch_status(api_key, batch_id)
                            if res.get("status") == "completed":
                                check = False
                                break
                    if not check:
                        res = await get_batch_executions(api_key, batch_id)
                        print(res)
                        return res


if __name__ == "__main__":
    asyncio.run(create_batch())
```

**Response:**
200 OK
```json
{
  "batch_id": "uuid-string",
  "state": "created"
}
```

### Get All Agents
Retrieves all agents from the system.

**Endpoint:** `GET /all`

**Response:**
200 OK
```json
{
  "agents": [
    {
      "agent_id": "string",
      "data": {
        "agent_config": {
          "agent_name": "Alfred",
          "agent_type": "other",
          "tasks": []
        },
        "agent_prompts": {}
      }
    }
  ]
}
