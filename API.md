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
```

---

## Bring Your Own Trunk (BYOT) - SIP Trunk Endpoints

For the full integration guide including provider setup, authentication methods, call lifecycle, and troubleshooting, see [docs/bring-your-own-trunk.md](docs/bring-your-own-trunk.md).

### Create SIP Trunk

Creates a new SIP trunk and registers it with Bolna's media layer.

**Endpoint:** `POST /sip-trunks/trunks`

**Request Body:**
```json
{
  "name": "My Production Trunk",
  "provider": "twilio",
  "description": "Main outbound trunk for US numbers",
  "auth_type": "userpass",
  "auth_username": "my_sip_username",
  "auth_password": "my_sip_password",
  "gateways": [
    {
      "gateway_address": "sip.example.com",
      "port": 5060,
      "priority": 1
    }
  ],
  "allow": "ulaw,alaw",
  "disallow": "all",
  "inbound_enabled": true,
  "outbound_leading_plus_enabled": true
}
```

**Parameters:**
- `name` (string, required) - Human-readable label for the trunk
- `provider` (string, required) - SIP provider name (e.g. `"twilio"`, `"plivo"`, `"zadarma"`)
- `description` (string) - Optional description
- `auth_type` (string, required) - `"userpass"` or `"ip-based"`
- `auth_username` (string) - Required when `auth_type` is `"userpass"`
- `auth_password` (string) - Required when `auth_type` is `"userpass"`
- `gateways` (array, required) - At least one gateway with `gateway_address`, `port`, `priority`
- `ip_identifiers` (array) - Required when `auth_type` is `"ip-based"`. List of `{ "ip_address": "..." }` objects
- `allow` (string) - Codecs to allow (default: `"ulaw,alaw"`)
- `disallow` (string) - Codecs to disallow (default: `"all"`)
- `inbound_enabled` (boolean) - Enable inbound calls (default: `false`)
- `outbound_leading_plus_enabled` (boolean) - Prepend `+` to outbound numbers (default: `true`)
- `phone_numbers` (array) - Optional phone numbers to add at creation time

**Response:**
201 Created
```json
{
  "id": "01HQXYZ123ABC456DEF",
  "user_id": "user_abc123",
  "name": "My Production Trunk",
  "provider": "twilio",
  "auth_type": "userpass",
  "gateways": [...],
  "phone_numbers": [],
  "is_active": true,
  "created_at": "2025-01-15T10:00:00Z",
  "updated_at": "2025-01-15T10:00:00Z"
}
```

### List All SIP Trunks

**Endpoint:** `GET /sip-trunks/trunks`

**Query Parameters:**
- `is_active` (boolean) - Filter by active status

### Get SIP Trunk

**Endpoint:** `GET /sip-trunks/trunks/{trunk_id}`

**Parameters:**
- `trunk_id` (path, required) - Trunk ID

### Update SIP Trunk

Partially updates an existing trunk. Only included fields are changed.

**Endpoint:** `PATCH /sip-trunks/trunks/{trunk_id}`

**Parameters:**
- `trunk_id` (path, required) - Trunk ID

**Request Body:** Any subset of trunk fields. Note: `gateways` and `ip_identifiers` arrays are fully replaced when provided.

**Response:**
200 OK - Returns the full updated trunk object.

### Delete SIP Trunk

Permanently deletes a trunk and all associated resources (gateways, IP identifiers, phone numbers).

**Endpoint:** `DELETE /sip-trunks/trunks/{trunk_id}`

**Parameters:**
- `trunk_id` (path, required) - Trunk ID

**Response:**
200 OK
```json
{
  "message": "Trunk deleted successfully. 3 phone number(s) removed."
}
```

### Add Phone Number to Trunk

Associates a DID phone number with a trunk.

**Endpoint:** `POST /sip-trunks/trunks/{trunk_id}/numbers`

**Parameters:**
- `trunk_id` (path, required) - Trunk ID

**Request Body:**
```json
{
  "phone_number": "919876543210",
  "name": "Mumbai Support Line",
  "e164_check_enabled": false
}
```

**Parameters:**
- `phone_number` (string, required) - Phone number with or without `+` prefix
- `name` (string) - Human-readable label
- `e164_check_enabled` (boolean) - Validate E.164 format (default: `false`)
- `is_active` (boolean) - Whether the number is active (default: `true`)

**Response:**
201 Created
```json
{
  "id": "01HQNUMBER111222333",
  "phone_number": "919876543210",
  "byot_trunk_id": "01HQXYZ123ABC456DEF",
  "telephony_provider": "sip-trunk",
  "created_at": "2025-01-15T10:05:00Z"
}
```

### List Phone Numbers on Trunk

**Endpoint:** `GET /sip-trunks/trunks/{trunk_id}/numbers`

**Parameters:**
- `trunk_id` (path, required) - Trunk ID

### Remove Phone Number from Trunk

**Endpoint:** `DELETE /sip-trunks/trunks/{trunk_id}/numbers/{phone_number_id}`

**Parameters:**
- `trunk_id` (path, required) - Trunk ID
- `phone_number_id` (path, required) - Phone number ID

**Response:**
200 OK
```json
{
  "message": "Phone number removed successfully"
}
```

### Set Up Inbound Call Routing

Maps a phone number to an agent for inbound calls.

**Endpoint:** `POST /inbound/setup`

**Request Body:**
```json
{
  "agent_id": "your-agent-uuid",
  "phone_number_id": "01HQNUMBER111222333"
}
```

**Parameters:**
- `agent_id` (string, required) - UUID of the Bolna agent
- `phone_number_id` (string, required) - Phone number ID from the trunk

**Response:**
200 OK
```json
{
  "message": "SIP trunk number successfully mapped to agent"
}
```

### Unlink Phone Number from Agent

Removes the agent mapping from a phone number without deleting the number from the trunk.

**Endpoint:** `POST /inbound/unlink`

**Request Body:**
```json
{
  "agent_id": "your-agent-uuid",
  "phone_number_id": "01HQNUMBER111222333"
}
```

**Parameters:**
- `agent_id` (string, required) - UUID of the Bolna agent
- `phone_number_id` (string, required) - Phone number ID
