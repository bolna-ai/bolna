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
