# Bring Your Own Trunk (BYOT) - SIP Trunk Integration Guide

Bolna supports connecting your own SIP trunks to the platform, allowing you to use your existing telephony relationships and numbers for AI-powered voice conversations. This guide covers every step: provisioning the trunk, adding phone numbers, mapping numbers to agents for inbound calls, and making outbound calls.

---

## Table of Contents

1. [Overview](#overview)
2. [SIP Provider Setup (Prerequisites)](#sip-provider-setup-prerequisites)
3. [Step 1 - Create a SIP Trunk](#step-1--create-a-sip-trunk)
4. [Step 2 - Add Phone Numbers to Your Trunk](#step-2--add-phone-numbers-to-your-trunk)
5. [Step 3 - Set Up Inbound Calls (Map a DID to an Agent)](#step-3--set-up-inbound-calls-map-a-did-to-an-agent)
6. [Step 4 - Make Outbound Calls](#step-4--make-outbound-calls)
7. [Managing Your Trunk](#managing-your-trunk)
   - [List All Trunks](#list-all-trunks)
   - [Get a Single Trunk](#get-a-single-trunk)
   - [List Numbers on a Trunk](#list-numbers-on-a-trunk)
   - [Update a Trunk](#update-a-trunk)
   - [Remove a Phone Number from a Trunk](#remove-a-phone-number-from-a-trunk)
   - [Unlink a Number from an Agent](#unlink-a-number-from-an-agent)
   - [Delete a Trunk](#delete-a-trunk)
8. [Authentication Methods Reference](#authentication-methods-reference)
9. [Parameter Reference](#parameter-reference)
10. [Call Lifecycle and Status Values](#call-lifecycle-and-status-values)
11. [Common Errors and Troubleshooting](#common-errors-and-troubleshooting)

---

## Overview

**Bring Your Own Trunk (BYOT)** lets you connect any standards-compliant SIP trunk to the Bolna platform. Once connected, Bolna can:

- **Receive inbound calls** on your DID numbers and route them to AI agents.
- **Place outbound calls** from your DID numbers using your trunk's minutes and rates.

Your trunk credentials and gateway addresses are stored securely. Bolna configures its internal Asterisk media layer automatically -- you do not need to manage any Asterisk configuration yourself.

> **Current limitation:** SRTP (Secure RTP) is **not supported**. Media must be transmitted over standard RTP. Ensure your SIP trunk is configured to use unencrypted RTP. Trunks that require mandatory SRTP will not work with Bolna at this time.

**Supported authentication methods:**

| Method | How it works |
|--------|-------------|
| `userpass` | Bolna authenticates to your SIP trunk using a username and password (SIP REGISTER or INVITE auth). |
| `ip-based` | Your SIP trunk identifies Bolna by source IP address; no username/password required. You must whitelist Bolna's IP on your trunk. |

---

## SIP Provider Setup (Prerequisites)

Before creating a trunk in Bolna, configure the following on your SIP trunk provider's portal:

### 1. Whitelist Bolna's IP Address

Bolna's SIP media server IP is:

```
13.200.45.61
```

You **must** whitelist this IP on your SIP trunk so that Bolna's outbound SIP INVITE and RTP packets are accepted. In most provider portals this is called an **IP whitelist**, **allowed IP**, **trusted IP**, or **ACL**.

If you are using `ip-based` authentication, this is the IP you will add as your identifier -- Bolna will be recognized solely by its source IP, so no username/password is exchanged.

### 2. Set the Origination URL (for Inbound DID / Inbound Calls)

If you want calls to your DID numbers to ring through to Bolna, you need to point your SIP trunk's **origination URI** (also called **termination URI**, **inbound route**, or **SIP URI**) to:

```
sip:13.200.45.61:5060
```

In your provider portal this is usually labeled as:
- **Origination URI** or **Origination URL**
- **Inbound SIP URI**
- **SIP Termination Point**
- **Route to** / **Forward to**

Set this for every DID number (or for the trunk as a whole, depending on your provider) that you intend to route to Bolna. Incoming INVITEs to your DID numbers will then arrive at Bolna's Asterisk, which matches the DID against your registered phone numbers and routes the call to the correct agent.

> **Note:** If your provider uses a **domain-based** origination URI rather than an IP, contact Bolna support for an alternative hostname.

### 3. Codec Configuration

Bolna's SIP layer uses **G.711 u-law (ulaw)** audio by default. Ensure your trunk allows ulaw, or at minimum alaw, in its codec preferences. G.729 and other compressed codecs are not recommended.

### 4. Disable SRTP on Your Trunk

**SRTP (Secure RTP) is not currently supported by Bolna.** Media is transmitted over standard (unencrypted) RTP. You must configure your SIP trunk to use plain RTP for media. If your provider has SRTP enabled by default or as a mandatory setting, disable it before connecting the trunk to Bolna -- calls will fail to establish media if SRTP is required by either side.

---

## Step 1 - Create a SIP Trunk

**Endpoint:** `POST /sip-trunks/trunks`

**Authentication:** API key required (`Authorization: Bearer <your-api-key>`)

This creates the trunk, registers it with Bolna's Asterisk media layer, and returns a trunk ID you will use in subsequent calls.

### Request Body

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

### Field Reference - Create Trunk

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | **Yes** | -- | A human-readable label for the trunk. Must be unique per account. |
| `provider` | string | **Yes** | -- | The SIP provider name. Free-form text (e.g. `"twilio"`, `"plivo"`, `"zadarma"`, `"telnyx"`, `"vonage"`, `"custom"`). Used for your own reference and reporting -- does not affect routing logic. |
| `description` | string | No | `null` | Optional free-text description for internal reference. |
| `auth_type` | string | **Yes** | -- | Authentication method. Must be `"userpass"` or `"ip-based"`. See [Authentication Methods Reference](#authentication-methods-reference). |
| `auth_username` | string | Conditional | `null` | SIP authentication username. **Required when `auth_type` is `"userpass"`**. |
| `auth_password` | string | Conditional | `null` | SIP authentication password. **Required when `auth_type` is `"userpass"`**. |
| `gateways` | array | **Yes** | -- | One or more SIP gateway addresses. At least one is required. See [Gateway Object](#gateway-object). |
| `ip_identifiers` | array | Conditional | `[]` | List of IP addresses or CIDR ranges that identify your trunk. **Required when `auth_type` is `"ip-based"`**. See [IP Identifier Object](#ip-identifier-object). |
| `allow` | string | No | `"ulaw,alaw"` | Comma-separated list of codecs to allow on this trunk. Bolna's media layer uses ulaw internally; always include `ulaw`. |
| `disallow` | string | No | `"all"` | Comma-separated list of codecs to disallow. Setting this to `"all"` combined with an explicit `allow` list is the recommended pattern. |
| `inbound_enabled` | boolean | No | `false` | Set to `true` if you plan to receive inbound calls through this trunk. This flag signals Bolna to configure the Asterisk dialplan context to accept incoming INVITEs. |
| `outbound_leading_plus_enabled` | boolean | No | `true` | When `true`, Bolna prepends a `+` to the dialed number in the SIP INVITE Request-URI. Most carriers expect E.164 format (`+919876543210`). Set to `false` only if your carrier does not accept a leading `+`. |
| `transport` | string | No | `"transport-udp"` | SIP transport protocol. Usually `"transport-udp"`. Use `"transport-tls"` only if your trunk requires TLS signalling and your Bolna setup supports it. |
| `direct_media` | boolean | No | `false` | When `true`, Bolna attempts to set up RTP directly between the caller and callee, bypassing Bolna's media server. Keep `false` unless specifically directed by support. |
| `rtp_symmetric` | boolean | No | `true` | Enables symmetric RTP (send and receive on the same port). Required by most NAT environments. Strongly recommended to leave as `true`. |
| `force_rport` | boolean | No | `true` | Forces Bolna to send responses to the source port of the request. Required for NAT traversal. Strongly recommended to leave as `true`. |
| `ice_support` | boolean | No | `true` | Enables ICE (Interactive Connectivity Establishment) for media negotiation. Recommended to leave as `true`. |
| `qualify_frequency` | integer | No | `60` | How often (in seconds) Bolna sends a SIP OPTIONS ping to the gateway to check its availability. Set to `0` to disable. |
| `phone_numbers` | array | No | `[]` | Optional list of phone numbers to associate with the trunk at creation time. See [Phone Number Create Object](#phone-number-create-object). You can also add numbers later via the dedicated endpoint. |

#### Gateway Object

Each item in the `gateways` array describes one SIP gateway:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `gateway_address` | string | **Yes** | -- | Hostname or IP address of the SIP gateway (e.g. `"sip.twilio.com"`, `"208.68.165.52"`). |
| `port` | integer | No | `5060` | SIP signalling port on the gateway. Standard is `5060` for UDP/TCP, `5061` for TLS. |
| `priority` | integer | No | `1` | Gateway selection priority. Lower number = higher priority. If you provide multiple gateways, Bolna uses the lowest-priority one as the primary and others as failover. |

> **Multiple gateways:** You can provide multiple gateways to enable failover. If the primary gateway is unreachable, Asterisk will attempt the next one in priority order.

#### IP Identifier Object

Each item in the `ip_identifiers` array identifies a source IP or network that should be associated with this trunk (used when `auth_type` is `"ip-based"`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ip_address` | string | **Yes** | IPv4 address or CIDR range (e.g. `"203.0.113.5"`, `"203.0.113.0/24"`). Incoming SIP traffic from these addresses will be matched to this trunk. |

#### Phone Number Create Object (optional at trunk creation)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `phone_number` | string | **Yes** | -- | The phone number in E.164 format, or without the `+` prefix (e.g. `"919876543210"` or `"+919876543210"`). |
| `name` | string | No | `null` | A label for the number (e.g. `"Support Line"`). |
| `e164_check_enabled` | boolean | No | `false` | When `true`, Bolna validates that the number is in strict E.164 format before saving. |
| `is_active` | boolean | No | `true` | Whether this number is active. |

### Example - Username/Password Authentication

```bash
curl -X POST https://api.bolna.ai/sip-trunks/trunks \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Zadarma Production",
    "provider": "zadarma",
    "description": "Main Zadarma trunk for outbound sales",
    "auth_type": "userpass",
    "auth_username": "bolnaTesting",
    "auth_password": "SuperSecurePassword123",
    "gateways": [
      {
        "gateway_address": "sip.zadarma.com",
        "port": 5060,
        "priority": 1
      }
    ],
    "allow": "ulaw,alaw",
    "disallow": "all",
    "inbound_enabled": true,
    "outbound_leading_plus_enabled": true
  }'
```

### Example - IP-Based Authentication (e.g. Plivo Elastic SIP)

```bash
curl -X POST https://api.bolna.ai/sip-trunks/trunks \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Plivo Elastic Trunk",
    "provider": "plivo",
    "auth_type": "ip-based",
    "gateways": [
      {
        "gateway_address": "21467306465797919.zt.plivo.com",
        "port": 5060,
        "priority": 1
      }
    ],
    "ip_identifiers": [
      { "ip_address": "15.207.90.192/31" },
      { "ip_address": "204.89.151.128/27" },
      { "ip_address": "13.52.9.0/25" }
    ],
    "inbound_enabled": true
  }'
```

### Success Response (`201 Created`)

```json
{
  "id": "01HQXYZ123ABC456DEF",
  "user_id": "user_abc123",
  "name": "Zadarma Production",
  "provider": "zadarma",
  "description": "Main Zadarma trunk for outbound sales",
  "auth_type": "userpass",
  "auth_username": "bolnaTesting",
  "gateways": [
    {
      "id": "01HQXYZ111AAA111BBB",
      "gateway_address": "sip.zadarma.com",
      "port": 5060,
      "priority": 1,
      "created_at": "2025-01-15T10:00:00Z"
    }
  ],
  "ip_identifiers": [],
  "phone_numbers": [],
  "allow": "ulaw,alaw",
  "disallow": "all",
  "transport": "transport-udp",
  "direct_media": false,
  "rtp_symmetric": true,
  "force_rport": true,
  "ice_support": true,
  "qualify_frequency": 60,
  "inbound_enabled": true,
  "outbound_leading_plus_enabled": true,
  "is_active": true,
  "created_at": "2025-01-15T10:00:00Z",
  "updated_at": "2025-01-15T10:00:00Z"
}
```

> Save the `id` field -- this is your **trunk ID** used in all subsequent requests.

---

## Step 2 - Add Phone Numbers to Your Trunk

**Endpoint:** `POST /sip-trunks/trunks/{trunk_id}/numbers`

**Authentication:** API key required

Associates a DID (phone number) you own with this trunk. Bolna stores the number and can route inbound calls to it once you map it to an agent.

### Path Parameters

| Parameter | Description |
|-----------|-------------|
| `trunk_id` | The trunk ID returned when you created the trunk. |

### Request Body

```json
{
  "phone_number": "919876543210",
  "name": "Mumbai Support Line",
  "e164_check_enabled": false
}
```

### Field Reference - Add Phone Number

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `phone_number` | string | **Yes** | -- | The phone number to add. Can be provided with or without the leading `+` (e.g. `"919876543210"` or `"+919876543210"`). This must be a number you own and have pointed to Bolna's origination URI (`sip:13.200.45.61:5060`) at your provider. |
| `name` | string | No | `null` | A human-readable label for this number (e.g. `"Support Line"`, `"Sales Mumbai"`). |
| `e164_check_enabled` | boolean | No | `false` | When `true`, validates that the number is in strict E.164 format (`+` followed by country code and subscriber number). Set to `false` to allow numbers without the `+` prefix. |
| `is_active` | boolean | No | `true` | Whether the number is active. |

> **Note on phone number format:** Bolna stores the number exactly as provided. When matching inbound calls, the platform performs a flexible lookup that checks both the number with and without a `+` prefix. Use a consistent format across your trunk and inbound DID configuration.

### Example

```bash
curl -X POST https://api.bolna.ai/sip-trunks/trunks/01HQXYZ123ABC456DEF/numbers \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "919876543210",
    "name": "Mumbai Support Line"
  }'
```

### Success Response (`201 Created`)

```json
{
  "id": "01HQNUMBER111222333",
  "phone_number": "919876543210",
  "user_id": "user_abc123",
  "byot_trunk_id": "01HQXYZ123ABC456DEF",
  "telephony_provider": "sip-trunk",
  "deleted": false,
  "created_at": "2025-01-15T10:05:00Z",
  "updated_at": "2025-01-15T10:05:00Z"
}
```

> Save the `id` field -- this is the **phone number ID** used when setting up inbound call routing.

---

## Step 3 - Set Up Inbound Calls (Map a DID to an Agent)

**Endpoint:** `POST /inbound/setup`

**Authentication:** API key required

Maps a phone number to a Bolna agent so that when a call arrives on that DID, the agent answers it. Before calling this endpoint, ensure:

1. The phone number has been added to the trunk (Step 2).
2. Your SIP provider is routing that DID to `sip:13.200.45.61:5060`.
3. You have an agent created in Bolna.

### Request Body

```json
{
  "agent_id": "your-agent-uuid",
  "phone_number_id": "01HQNUMBER111222333"
}
```

### Field Reference - Inbound Setup

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_id` | string | **Yes** | The UUID of the Bolna agent that should handle calls on this number. |
| `phone_number_id` | string | **Yes** | The phone number ID returned when you added the number to the trunk (Step 2). |

### Example

```bash
curl -X POST https://api.bolna.ai/inbound/setup \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agt-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "phone_number_id": "01HQNUMBER111222333"
  }'
```

### Success Response (`200 OK`)

```json
{
  "message": "SIP trunk number successfully mapped to agent"
}
```

Once this is done, any call arriving on that DID will be answered by the specified agent using your SIP trunk. The agent's audio format is automatically updated to `ulaw` to match Asterisk's requirements.

> **One number, one agent:** A phone number can only be mapped to one agent at a time. Mapping it to a new agent automatically unmaps it from the previous one.

---

## Step 4 - Make Outbound Calls

Outbound calls from your BYOT trunk are placed via the standard Bolna call API. The key requirement is that your agent is configured to use the `sip-trunk` telephony provider and your agent's outbound number is one of the DIDs you registered on the trunk.

### Configure the Agent's Telephony Provider

When creating or updating your agent, set `telephony_provider` to `"sip-trunk"` in the `agent_config`. This tells Bolna to route outbound calls through your SIP trunk and use ulaw audio encoding.

```bash
curl -X PATCH https://api.bolna.ai/{agent_id} \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_config": {
      "telephony_provider": "sip-trunk"
    }
  }'
```

### Place an Outbound Call

Use the standard call initiation endpoint, specifying the `from_number` as a DID registered on your trunk:

```bash
curl -X POST https://api.bolna.ai/call \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agt-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "recipient": {
      "phone_number": "+918800001234",
      "name": "Rahul Sharma"
    },
    "from_number": "+919876543210"
  }'
```

Bolna looks up the `from_number` in the registered phone numbers table, resolves the associated trunk and gateway, and places the call via Asterisk's PJSIP channel driver using your trunk's credentials.

---

## Managing Your Trunk

### List All Trunks

**Endpoint:** `GET /sip-trunks/trunks`

Returns all SIP trunks for your account, along with the count of phone numbers associated with each.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `is_active` | boolean | No | Filter by active status. Omit to return all trunks. |

```bash
curl https://api.bolna.ai/sip-trunks/trunks \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx"

# Only active trunks
curl "https://api.bolna.ai/sip-trunks/trunks?is_active=true" \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx"
```

### Get a Single Trunk

**Endpoint:** `GET /sip-trunks/trunks/{trunk_id}`

Returns full trunk details including all associated gateways, IP identifiers, and phone numbers.

```bash
curl https://api.bolna.ai/sip-trunks/trunks/01HQXYZ123ABC456DEF \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx"
```

### List Numbers on a Trunk

**Endpoint:** `GET /sip-trunks/trunks/{trunk_id}/numbers`

Returns all phone numbers associated with a trunk, including which agent (if any) each number is mapped to for inbound calls.

```bash
curl https://api.bolna.ai/sip-trunks/trunks/01HQXYZ123ABC456DEF/numbers \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx"
```

---

### Update a Trunk

**Endpoint:** `PATCH /sip-trunks/trunks/{trunk_id}`

**Authentication:** API key required

Partially updates an existing trunk. Only the fields you include in the request body are changed; all other fields remain unchanged.

> **Gateway and IP identifier replacement:** If you include `gateways` or `ip_identifiers` in a PATCH request, the entire list is **replaced** (not merged). To keep existing entries, include them in the new list.

### Request Body

All fields are optional:

```json
{
  "name": "Updated Trunk Name",
  "description": "Updated description",
  "is_active": true,
  "allow": "ulaw",
  "disallow": "all",
  "auth_username": "new_username",
  "auth_password": "new_password",
  "qualify_frequency": 30,
  "inbound_enabled": true,
  "outbound_leading_plus_enabled": false,
  "gateways": [
    {
      "gateway_address": "sip.newgateway.com",
      "port": 5060,
      "priority": 1
    }
  ],
  "ip_identifiers": [
    { "ip_address": "203.0.113.10" }
  ],
  "direct_media": false,
  "rtp_symmetric": true,
  "force_rport": true,
  "ice_support": true
}
```

### Field Reference - Update Trunk

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | New trunk name. |
| `description` | string | New trunk description. |
| `is_active` | boolean | Enable or disable the trunk. Setting to `false` prevents new calls from using this trunk but does not affect active calls. |
| `allow` | string | Replace the allowed codec list (e.g. `"ulaw,alaw"`). |
| `disallow` | string | Replace the disallowed codec list. |
| `auth_username` | string | New SIP authentication username (only for `userpass` trunks). |
| `auth_password` | string | New SIP authentication password (only for `userpass` trunks). Stored securely and never returned in GET responses. |
| `qualify_frequency` | integer | New SIP OPTIONS ping interval in seconds. |
| `inbound_enabled` | boolean | Enable or disable inbound call acceptance on this trunk. |
| `outbound_leading_plus_enabled` | boolean | Enable or disable `+` prefix on outbound dialed numbers. |
| `gateways` | array | **Full replacement.** Provide the complete new list of gateways. Same format as during creation. |
| `ip_identifiers` | array | **Full replacement.** Provide the complete new list of IP identifiers. Same format as during creation. |
| `direct_media` | boolean | Update direct media setting. |
| `rtp_symmetric` | boolean | Update symmetric RTP setting. |
| `force_rport` | boolean | Update force rport setting. |
| `ice_support` | boolean | Update ICE support setting. |

### Example - Update Gateway and Disable Outbound Leading Plus

```bash
curl -X PATCH https://api.bolna.ai/sip-trunks/trunks/01HQXYZ123ABC456DEF \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "outbound_leading_plus_enabled": false,
    "gateways": [
      {
        "gateway_address": "sip.newprovider.com",
        "port": 5060,
        "priority": 1
      }
    ]
  }'
```

### Example - Rotate Credentials

```bash
curl -X PATCH https://api.bolna.ai/sip-trunks/trunks/01HQXYZ123ABC456DEF \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "auth_username": "new_bolna_user",
    "auth_password": "NewSecurePassword456"
  }'
```

### Success Response (`200 OK`)

Returns the full updated trunk object (same format as the create response).

---

### Remove a Phone Number from a Trunk

**Endpoint:** `DELETE /sip-trunks/trunks/{trunk_id}/numbers/{phone_number_id}`

Removes a phone number from a trunk. If the number was mapped to an agent for inbound calls, the mapping is also removed.

```bash
curl -X DELETE \
  https://api.bolna.ai/sip-trunks/trunks/01HQXYZ123ABC456DEF/numbers/01HQNUMBER111222333 \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx"
```

**Success Response (`200 OK`):**

```json
{
  "message": "Phone number removed successfully"
}
```

---

### Unlink a Number from an Agent

**Endpoint:** `POST /inbound/unlink`

Removes the agent mapping from a phone number without deleting the number from the trunk. After unlinking, inbound calls to that number will no longer be answered by an agent (they will be rejected at the SIP level).

```bash
curl -X POST https://api.bolna.ai/inbound/unlink \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agt-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "phone_number_id": "01HQNUMBER111222333"
  }'
```

---

### Delete a Trunk

**Endpoint:** `DELETE /sip-trunks/trunks/{trunk_id}`

Permanently deletes a trunk and all associated resources:

- All gateways for the trunk are removed.
- All IP identifiers are removed.
- All phone numbers are soft-deleted and their agent mappings are cleared.
- The Asterisk PJSIP configuration is removed from Bolna's media layer.

> **Warning:** Deleting a trunk while calls are in progress may cause those calls to be interrupted. Deactivate the trunk first (`PATCH` with `"is_active": false`) and wait for active calls to finish before deleting.

```bash
curl -X DELETE https://api.bolna.ai/sip-trunks/trunks/01HQXYZ123ABC456DEF \
  -H "Authorization: Bearer bn-xxxxxxxxxxxxxxxx"
```

**Success Response (`200 OK`):**

```json
{
  "message": "Trunk deleted successfully. 3 phone number(s) removed."
}
```

---

## Authentication Methods Reference

### Username/Password (`userpass`)

Bolna acts as a SIP UAC (User Agent Client) and authenticates to your trunk using standard SIP digest authentication. You provide a username and password; Bolna includes these credentials in REGISTER or INVITE requests.

**Required fields:** `auth_username`, `auth_password`

**When to use:** Most hosted SIP trunks (Twilio Elastic SIP, Zadarma, Vonage, etc.) offer credential-based auth as the default.

**Your provider setup:** Create a SIP endpoint or trunk credential set on your provider portal and supply those credentials when creating the Bolna trunk.

---

### IP-Based (`ip-based`)

Your SIP trunk identifies and trusts incoming traffic from Bolna's server by IP address alone. No SIP username or password is exchanged.

**Required fields:** `ip_identifiers` (list of IP addresses or CIDR ranges)

**Required provider setup:** Add Bolna's IP `13.200.45.61` to your trunk's IP whitelist or allowed IP list on your provider portal.

**When to use:** Providers like Plivo Elastic SIP, DIDWW, and some enterprise trunks support IP-based authentication. This is often more reliable in high-volume environments because there is no SIP REGISTER overhead.

**`ip_identifiers` in the Bolna request:** These are the IPs that Bolna's Asterisk will use when matching inbound SIP traffic to your trunk. If your provider sends traffic from multiple IP ranges, add all of them. Example for Plivo:

```json
"ip_identifiers": [
  { "ip_address": "15.207.90.192/31" },
  { "ip_address": "204.89.151.128/27" },
  { "ip_address": "13.52.9.0/25" }
]
```

---

## Parameter Reference

### Complete Trunk Object (as returned by the API)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique trunk identifier (ULID format). |
| `user_id` | string | The user account this trunk belongs to. |
| `name` | string | Human-readable trunk name. |
| `provider` | string | Provider name (free-form, for reference). |
| `description` | string or null | Optional description. |
| `auth_type` | string | `"userpass"` or `"ip-based"`. |
| `auth_username` | string or null | SIP username (returned for reference; password is never returned). |
| `gateways` | array | List of gateway objects. |
| `ip_identifiers` | array | List of IP identifier objects. |
| `phone_numbers` | array | List of phone number objects associated with this trunk. |
| `allow` | string | Allowed codec list. |
| `disallow` | string | Disallowed codec list. |
| `transport` | string | SIP transport (`"transport-udp"`, etc.). |
| `direct_media` | boolean | Direct media flag. |
| `rtp_symmetric` | boolean | Symmetric RTP flag. |
| `force_rport` | boolean | Force rport flag. |
| `ice_support` | boolean | ICE support flag. |
| `qualify_frequency` | integer | SIP OPTIONS ping interval in seconds. |
| `inbound_enabled` | boolean | Whether inbound calls are accepted. |
| `outbound_leading_plus_enabled` | boolean | Whether `+` is prepended to outbound dialed numbers. |
| `is_active` | boolean | Whether the trunk is currently active. |
| `created_at` | string | ISO 8601 timestamp of creation. |
| `updated_at` | string | ISO 8601 timestamp of last update. |

### Complete Phone Number Object (as returned by the API)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique phone number record ID (ULID format). Use this as `phone_number_id` in inbound setup and delete calls. |
| `phone_number` | string | The phone number as stored (E.164 or without `+`). |
| `user_id` | string | The user account this number belongs to. |
| `byot_trunk_id` | string | The trunk this number is associated with. |
| `telephony_provider` | string | Always `"sip-trunk"` for BYOT numbers. |
| `agent_id` | string or null | The agent currently mapped to this number for inbound calls. `null` if not mapped. |
| `deleted` | boolean | Whether this record has been soft-deleted. |
| `created_at` | string | ISO 8601 timestamp. |
| `updated_at` | string | ISO 8601 timestamp. |

---

## Call Lifecycle and Status Values

When a call is placed or received through your SIP trunk, Bolna tracks its status in the execution record. You can retrieve call statuses through the executions API.

### Execution Status Values

| Status | Description |
|--------|-------------|
| `initiated` | Outbound call has been submitted to the SIP trunk. |
| `ringing` | The remote party's phone is ringing. |
| `in-progress` | Call is active; AI agent and caller are connected. |
| `call-disconnected` | WebSocket disconnected; call cleanup in progress. |
| `completed` | Call ended normally with full data recorded. |
| `busy` | Remote party returned a busy signal. |
| `no-answer` | Remote party did not answer within the timeout. |
| `canceled` | Call was canceled before it was answered. |
| `failed` | Call failed due to a network or SIP error. |
| `balance-low` | Call was not attempted due to insufficient account balance. |

### Hangup Reasons

When a call ends, Bolna records the hangup reason mapped from the SIP cause codes:

| SIP Cause | Bolna Reason |
|-----------|-------------|
| Normal Clearing | Call ended normally |
| USER_BUSY | Call recipient was busy |
| NO_ANSWER | Call unanswered |
| CALL_REJECTED | Call recipient rejected |
| ORIGINATOR_CANCEL | Call canceled by originator |
| NORMAL_TEMPORARY_FAILURE | Temporary network failure |
| UNALLOCATED_NUMBER | Call recipient number invalid |
| NETWORK_OUT_OF_ORDER | Network error |
| INVALID_NUMBER_FORMAT | Invalid phone number format |

---

## Common Errors and Troubleshooting

### `"Trunk name is required"` (400)
The `name` field was missing or empty in the create trunk request.

### `"Provider is required"` (400)
The `provider` field was missing in the create trunk request.

### `"auth_type must be 'userpass' or 'ip-based'"` (400)
The `auth_type` field was missing or contained an invalid value.

### `"At least one gateway is required"` (400)
The `gateways` array was empty or missing.

### `"auth_username and auth_password are required for userpass auth"` (400)
When `auth_type` is `"userpass"`, both `auth_username` and `auth_password` must be provided.

### `"ip_identifiers are required for ip-based auth"` (400)
When `auth_type` is `"ip-based"`, at least one IP identifier must be provided.

### `"SIP trunk phone number not found in database"` (404)
You tried to map a SIP trunk number for inbound that has not been added to the trunk yet. Complete Step 2 (Add Phone Number) before calling the inbound setup endpoint.

### `"Phone number doesn't exist."` (404)
The `phone_number_id` provided in the inbound setup request does not exist or does not belong to your account.

### `"Phone number already exists on another active trunk"` (409 / 422)
The phone number you are trying to add is already registered on a different active trunk. Remove it from the other trunk first before adding it here.

### Inbound calls not arriving / call rejected by Bolna

1. Confirm your SIP provider has `sip:13.200.45.61:5060` set as the origination URI for the DID.
2. Confirm the phone number has been added to the trunk in Bolna (Step 2).
3. Confirm the phone number has been mapped to an agent (Step 3).
4. Confirm `inbound_enabled` is `true` on the trunk.
5. Check that the number format in the INVITE's `Request-URI` or `To` header matches exactly what you stored (with or without `+`).

### Outbound calls failing / calls not placed

1. Confirm `is_active` is `true` on the trunk.
2. Confirm the `from_number` used in the call request is registered on the trunk.
3. Confirm the gateway address and credentials are correct. Check your provider portal for any authentication errors.
4. For `ip-based` trunks, ensure `13.200.45.61` is in your provider's IP whitelist.
5. Confirm `outbound_leading_plus_enabled` matches what your carrier expects (some carriers reject `+`, others require it).

### No audio / call connects but no voice (SRTP mismatch)

SRTP is not supported. If your SIP trunk has SRTP enabled (mandatory or preferred), the media negotiation will fail and the call will connect at the SIP signalling level but carry no audio. Disable SRTP on your trunk and ensure media is negotiated as plain RTP.

### Audio quality issues / one-way audio

- Ensure `rtp_symmetric` and `force_rport` are both `true` (the defaults). These are essential for NAT traversal.
- Confirm your SIP provider's RTP IP ranges are reachable from `13.200.45.61`.
- Verify `allow` includes `ulaw` -- Bolna's media layer uses G.711 u-law internally.
- Avoid enabling `direct_media` unless explicitly advised by Bolna support, as it bypasses the media server required for AI audio processing.

### SIP OPTIONS / qualify failing

If `qualify_frequency` is set but the gateway is not responding to OPTIONS, it may indicate a firewall blocking UDP 5060 from `13.200.45.61` to your gateway. Check your gateway's firewall rules.

---

*For additional assistance, contact Bolna support or visit [docs.bolna.ai](https://docs.bolna.ai).*
