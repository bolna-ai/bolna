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

#### Resemble Detect and Signal API tool example

Bolna agents can call external HTTP APIs through `tools_config.api_tools`. Use
this pattern when the caller provides a media URL that should be checked for
synthetic speech or manipulated audio, or when caller-provided text should be
scored for fraud or scam intent before the agent makes a trust decision.

Add the tool definition inside a task's `tools_config`:

```json
{
  "api_tools": {
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "resemble_submit_detection",
          "description": "Submit a public HTTPS media URL to Resemble Detect for synthetic speech or manipulated media analysis. Use this when a caller provides a recording URL or another media artifact that should be verified.",
          "parameters": {
            "type": "object",
            "properties": {
              "url": {
                "type": "string",
                "description": "Public HTTPS URL for the audio, image, or video to analyze."
              },
              "reason": {
                "type": "string",
                "description": "Short reason this media should be verified."
              }
            },
            "required": ["url"],
            "additionalProperties": false
          },
          "strict": true
        }
      },
      {
        "type": "function",
        "function": {
          "name": "resemble_score_signal",
          "description": "Score caller-provided text with Resemble Signal for fraud or scam intent. Use this before sharing reset codes, changing account details, initiating transfers, or acting on urgent payment requests.",
          "parameters": {
            "type": "object",
            "properties": {
              "text": {
                "type": "string",
                "description": "Caller-provided text or transcript excerpt to score."
              },
              "reason": {
                "type": "string",
                "description": "Short reason this request should be scored."
              }
            },
            "required": ["text"],
            "additionalProperties": false
          },
          "strict": true
        }
      }
    ],
    "tools_params": {
      "resemble_submit_detection": {
        "url": "https://app.resemble.ai/api/v2/detect",
        "method": "POST",
        "api_token": "Bearer <RESEMBLE_API_KEY>",
        "headers": {
          "Accept": "application/json",
          "Content-Type": "application/json"
        },
        "param": {
          "url": {
            "$var": "url"
          },
          "intelligence": true,
          "audio_source_tracing": true,
          "zero_retention_mode": true,
          "callback_url": "<OPTIONAL_DETECTION_RESULT_WEBHOOK_URL>"
        },
        "pre_call_message": "I will verify that media before making a trust decision."
      },
      "resemble_score_signal": {
        "url": "https://app.resemble.ai/api/v2/signal",
        "method": "POST",
        "api_token": "Bearer <RESEMBLE_API_KEY>",
        "headers": {
          "Accept": "application/json",
          "Content-Type": "application/json"
        },
        "param": {
          "text": {
            "$var": "text"
          }
        },
        "pre_call_message": "I will check whether this request matches a fraud pattern before continuing."
      }
    }
  }
}
```

Store the Resemble API key in your deployment secrets and inject it into
`api_token` at deploy time. If you use `callback_url`, handle the completed
Detect result in that webhook; the generic API tool executor sends one HTTP
request and does not poll `GET /detect/{uuid}` during the call. Signal text
scoring returns synchronously from `POST /signal`.

Suggested system prompt:

```text
When the caller provides a media URL that affects a trust or fraud decision,
call resemble_submit_detection before deciding. When the caller asks for a
reset code, credential, wire transfer, urgent payment, or account change, call
resemble_score_signal before deciding. Treat both responses as risk signals. If
Detect reports synthetic speech, watermark evidence, or suspicious source
tracing, or Signal returns suspicious/fraud, follow this agent's escalation
policy.
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
