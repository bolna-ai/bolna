# Sub-Accounts Documentation

## Overview

Sub-accounts in Bolna provide a powerful way to organize and manage multiple customer accounts or business units under a single main account. This feature enables you to maintain clear separation of Bolna agents, calls, logs, recordings, and usage data while centralizing management and billing.

Sub-accounts are owned by a parent account and can be used to segment, manage, and keep track of account usage independently. This organizational structure is particularly valuable for businesses that serve multiple clients, manage different departments, or need to track usage across various projects.

## Key Benefits

### Logical Separation
- **Isolated Workspaces**: Each sub-account operates as a separate workspace with dedicated configurations
- **Independent Agent Management**: Agents created in one sub-account are completely isolated from others
- **Separate Call Logs**: Call history and recordings are maintained separately for each sub-account
- **Usage Isolation**: Track consumption and costs independently for each sub-account

### Centralized Management
- **Single Dashboard**: Manage all sub-accounts from your main account dashboard
- **Unified Billing**: Consolidated billing across all sub-accounts while maintaining detailed usage breakdowns
- **Centralized Authentication**: Use your main account credentials to access and manage all sub-accounts

### Business Organization
- **Client Separation**: Create dedicated sub-accounts for each of your clients
- **Department Management**: Organize different business units or departments
- **Project Isolation**: Separate resources and usage by project or campaign
- **Multi-tenant Architecture**: Support multiple tenants with complete data isolation

## Use Cases

### Service Providers
If you're building voice AI solutions for multiple clients, sub-accounts allow you to:
- Create isolated environments for each client
- Track usage and costs per client for accurate billing
- Maintain separate agent configurations and call logs
- Provide clients with dedicated workspaces

### Enterprise Organizations
Large organizations can use sub-accounts to:
- Separate different departments (Sales, Support, Marketing)
- Manage regional offices independently
- Track usage across different business units
- Implement cost allocation and chargeback models

### Development Teams
Development teams can leverage sub-accounts for:
- Separating development, staging, and production environments
- Managing different projects or applications
- Tracking resource consumption per project
- Implementing proper access controls

## Getting Started

### Prerequisites
- Active Bolna account with enterprise features enabled
- Valid API authentication token
- Understanding of your organizational structure

### Basic Workflow
1. **Create Sub-Account**: Set up a new sub-account with a descriptive name
2. **Configure Agents**: Create and configure agents within the sub-account
3. **Make Calls**: Initiate calls using agents from the specific sub-account
4. **Track Usage**: Monitor consumption and costs for the sub-account
5. **Manage Resources**: Update configurations and manage resources as needed

## API Reference

All sub-account operations require authentication using your main account credentials. Include your API key in the `Authorization` header as `Bearer <token>`.

### Base URL
```
https://api.bolna.ai/sub-accounts
```

### Authentication
All requests must include the Authorization header:
```
Authorization: Bearer <your-api-token>
```

### Create Sub-Account

Create a new sub-account to define separate workspaces with dedicated configurations.

**Endpoint:** `POST /sub-accounts/create`

**Headers:**
```
Content-Type: application/json
Authorization: Bearer <your-api-token>
```

**Request Body:**
```json
{
  "name": "alpha-007",
  "multi_tenant": false,
  "db_host": "prod-alpha-007-db-east-1-av-database.com",
  "db_name": "alpha-007-db",
  "db_port": 5432,
  "db_user": "alpha-007_user",
  "db_password": "alpha-007_password"
}
```

**Parameters:**
- `name` (string, required): Name of the sub-account. Must be unique within your organization.
- `multi_tenant` (boolean, optional): Whether the sub-account supports multi-tenancy. Default: false
- `db_host` (string, optional): Database host for the sub-account
- `db_name` (string, optional): Database name for the sub-account
- `db_port` (integer, optional): Database port for the sub-account
- `db_user` (string, optional): Database user for the sub-account
- `db_password` (string, optional): Database password for the sub-account

**Response:**
```json
{
  "id": "3c903cc-0d4c-4b50-8888-8dd257360052a",
  "name": "alpha-007",
  "user_id": "3c903cc-0d4c-4b50-8888-8dd257360052a",
  "organization_id": "3c903cc-0d4c-4b50-8888-8dd257360052",
  "api_key": "sa-b33fcfdbf01d4661a23ae4b1018356f0",
  "multi_tenant": false,
  "db_host": null,
  "db_name": null,
  "db_port": null,
  "db_user": null,
  "db_password": null,
  "created_at": "2025-01-23T01:14:372",
  "updated_at": "2025-01-23T01:14:372"
}
```

### List Sub-Accounts

Retrieve all sub-accounts linked to your main account enabling centralized visibility and management.

**Endpoint:** `GET /sub-accounts/all`

**Headers:**
```
Authorization: Bearer <your-api-token>
```

**Response:**
```json
{
  "id": "3c903cc-0d4c-4b50-8888-8dd257360052a",
  "name": "alpha-007",
  "organization_id": "3c903cc-0d4c-4b50-8888-8dd257360052",
  "api_key": "sa-b33fcfdbf01d4661a23ae4b1018356f0",
  "multi_tenant": false
}
```

**Response Fields:**
- `id` (string): Unique identifier for the sub-account
- `name` (string): Name of the sub-account
- `organization_id` (string): ID of the parent organization
- `api_key` (string): API key for the sub-account
- `multi_tenant` (boolean): Whether multi-tenancy is enabled

### Track Sub-Account Usage

Track usage for a specific sub-account giving you fine-grained insights into usage, consumption and billing.

**Endpoint:** `GET /sub-accounts/{sub_account_id}/usage`

**Headers:**
```
Authorization: Bearer <your-api-token>
```

**Path Parameters:**
- `sub_account_id` (string, required): The unique identifier of the sub-account

**Query Parameters:**
- `from` (string, optional): The start timestamp in ISO format. If not provided, defaults to the start of the current month.

**Response:**
```json
{
  "from": "2025-06-25",
  "to": "2025-06-27T23:59:59.999999",
  "total_records": 12,
  "total_duration": 151,
  "total_cost": 31.754,
  "total_platform_cost": 22,
  "total_telephony_cost": 2.673,
  "status_map": {
    "completed": 11,
    "no-answer": 1
  },
  "synthesizer_cost_map": {
    "elevenlabs": {
      "elevenlabs_v2_5": {
        "characters": 577,
        "cost": 5.77
      }
    }
  },
  "transcriber_cost_map": {
    "deepgram": {
      "nova-2": {
        "duration": 125.5493724,
        "cost": 1.88
      }
    }
  },
  "llm_cost_map": {
    "cost": 0.078,
    "tokens": {
      "gpt-4o-mini": {
        "characters": 577,
        "cost": 5.77
      }
    }
  }
}
```

**Response Fields:**
- `from` (string): Start date of the usage period
- `to` (string): End date of the usage period
- `total_records` (integer): Total number of call records
- `total_duration` (integer): Total call duration in seconds
- `total_cost` (float): Total cost for the period
- `total_platform_cost` (float): Platform-specific costs
- `total_telephony_cost` (float): Telephony provider costs
- `status_map` (object): Breakdown of call statuses
- `synthesizer_cost_map` (object): Detailed cost breakdown by synthesizer
- `transcriber_cost_map` (object): Detailed cost breakdown by transcriber
- `llm_cost_map` (object): Detailed cost breakdown by LLM usage

### Get All Sub-Accounts Usage

Retrieve usage, consumption, and billing details for all sub-accounts under the authenticated user's organization.

**Endpoint:** `GET /sub-accounts/all/usage`

**Headers:**
```
Authorization: Bearer <your-api-token>
```

**Response:**
```json
{
  "sub_account_id": "019f3f5d-f4a1-427c-b1f8-6ba2b4731e4c",
  "sub_account_name": "alpha-sub-account",
  "from": "2025-06-25",
  "to": "2025-06-27T23:59:59.999999",
  "total_records": 12,
  "total_duration": 151,
  "total_cost": 31.754,
  "total_platform_cost": 22,
  "total_telephony_cost": 2.673,
  "status_map": {
    "completed": 11,
    "no-answer": 1
  },
  "synthesizer_cost_map": {
    "elevenlabs": {
      "elevenlabs_v2_5": {
        "characters": 577,
        "cost": 5.77
      }
    }
  },
  "transcriber_cost_map": {
    "deepgram": {
      "nova-2": {
        "duration": 125.5493724,
        "cost": 1.88
      }
    }
  },
  "llm_cost_map": {
    "cost": 0.078,
    "tokens": {
      "gpt-4o-mini": {
        "characters": 577,
        "cost": 5.77
      }
    }
  }
}
```

This endpoint returns aggregated usage data for all sub-accounts, providing a comprehensive view of your organization's consumption and costs.

## Examples

### Example 1: Creating a Client Sub-Account

```bash
curl -X POST https://api.bolna.ai/sub-accounts/create \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-token" \
  -d '{
    "name": "client-acme-corp",
    "multi_tenant": false
  }'
```

### Example 2: Tracking Monthly Usage

```bash
curl -X GET "https://api.bolna.ai/sub-accounts/3c903cc-0d4c-4b50-8888-8dd257360052a/usage?from=2025-01-01" \
  -H "Authorization: Bearer your-api-token"
```

### Example 3: Getting All Sub-Accounts

```bash
curl -X GET https://api.bolna.ai/sub-accounts/all \
  -H "Authorization: Bearer your-api-token"
```

## Best Practices

### Naming Conventions
- Use descriptive names that clearly identify the purpose or client
- Consider using prefixes for different types of sub-accounts (e.g., `client-`, `dept-`, `proj-`)
- Avoid special characters and spaces in sub-account names

### Security Considerations
- Each sub-account receives its own API key for secure access
- Regularly rotate API keys for enhanced security
- Implement proper access controls based on your organizational needs
- Monitor usage patterns to detect unusual activity

### Cost Management
- Regularly monitor usage across all sub-accounts
- Set up alerts for unusual consumption patterns
- Use the detailed cost breakdowns to optimize resource usage
- Implement cost allocation strategies based on your business model

### Operational Guidelines
- Document the purpose and ownership of each sub-account
- Establish clear processes for creating and managing sub-accounts
- Regularly review and clean up unused sub-accounts
- Maintain consistent configuration standards across sub-accounts

## Error Handling

### Common Error Responses

**400 Bad Request**
```json
{
  "error": "Invalid request parameters",
  "details": "Sub-account name already exists"
}
```

**401 Unauthorized**
```json
{
  "error": "Authentication failed",
  "details": "Invalid or expired API token"
}
```

**404 Not Found**
```json
{
  "error": "Sub-account not found",
  "details": "The specified sub-account ID does not exist"
}
```

**429 Too Many Requests**
```json
{
  "error": "Rate limit exceeded",
  "details": "Please wait before making additional requests"
}
```

## Integration with Bolna Agents

Once you have created sub-accounts, you can create and manage agents within each sub-account independently. Each agent will be associated with its parent sub-account, ensuring complete isolation of resources and data.

### Creating Agents in Sub-Accounts
When creating agents, use the sub-account's API key to ensure the agent is created within the correct sub-account context.

### Call Management
All calls initiated by agents within a sub-account are tracked and billed to that specific sub-account, providing clear cost attribution and usage tracking.

## Support and Resources

- **API Documentation**: [https://docs.bolna.ai/api-reference/sub-accounts/overview](https://docs.bolna.ai/api-reference/sub-accounts/overview)
- **Main Documentation**: [https://docs.bolna.ai](https://docs.bolna.ai)
- **Community Support**: [Discord](https://discord.gg/59kQWGgnm8)
- **Enterprise Support**: Contact your account manager or visit [bolna.ai](https://bolna.ai)

For additional help with sub-account implementation or advanced use cases, please reach out to our support team or join our community Discord server.
