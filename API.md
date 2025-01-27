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
{
  "agent_config": {
      "agent_name": "Alfred",
      "agent_type": "other",
      "agent_welcome_message": "How are you doing Bruce?",
      "tasks": [
          {
              "task_type": "conversation",
              "toolchain": {
                  "execution": "parallel",
                  "pipelines": [
                      [
                          "transcriber",
                          "llm",
                          "synthesizer"
                      ]
                  ]
              },
              "tools_config": {
                  "input": {
                      "format": "wav",
                      "provider": "twilio"
                  },
                  "llm_agent": {
                      "agent_type": "simple_llm_agent",
                      "agent_flow_type": "streaming",
                      "routes": null,
                      "llm_config": {
                          "agent_flow_type": "streaming",
                          "provider": "openai",
                          "request_json": true,
                          "model": "gpt-4o-mini"
                      }
                  },
                  "output": {
                      "format": "wav",
                      "provider": "twilio"
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
                  },
                  "transcriber": {
                      "encoding": "linear16",
                      "language": "en",
                      "provider": "deepgram",
                      "stream": true
                  }
              },
              "task_config": {
                  "hangup_after_silence": 30.0
              }
          }
      ]
  },
  "agent_prompts": {
      "task_1": {
          "system_prompt": "Why Do We Fall, Sir? So That We Can Learn To Pick Ourselves Up."
      }
  }
}



}
```

**Response:**
```json
200 OK
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
```json
200 OK
{
    "agent_id": "string",
    "state": "deleted"
}
```


### Get All Agents
Retrieves all agents from the system.

**Endpoint:** `GET /all`

**Response:**
```json
200 OK
{
    "agents": [
        {
            "agent_id": "string",
            "data": {
                // Agent configuration object
            }
        }
    ]
}
```