"""
TypedDict row types mirroring the PostgreSQL schema.

These are documentation / type-hint helpers only — asyncpg returns
asyncpg.Record objects which behave like dicts.  Cast with ``dict(row)``
or ``MyModel(**dict(row))`` as needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from typing_extensions import TypedDict


class OrgRow(TypedDict):
    id: UUID
    api_key: str
    name: str
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    max_concurrent_calls: int
    created_at: datetime
    updated_at: datetime


class AgentRow(TypedDict):
    id: UUID
    org_id: UUID
    name: str
    agent_config: dict[str, Any]   # JSONB
    created_at: datetime
    updated_at: datetime


class CallRow(TypedDict):
    id: UUID
    org_id: UUID
    agent_id: UUID
    call_sid: str
    direction: str           # "outbound" | "inbound"
    status: str
    from_number: str
    to_number: str
    duration_seconds: int | None
    recording_url: str | None
    transcript: list[dict] | None   # JSONB
    created_at: datetime
    updated_at: datetime


class CampaignRow(TypedDict):
    id: UUID
    org_id: UUID
    agent_id: UUID
    name: str
    status: str              # "created" | "running" | "paused" | "completed" | "failed"
    contact_list_id: UUID | None
    template_variables: dict[str, Any]  # JSONB
    total_contacts: int
    dialed: int
    answered: int
    created_at: datetime
    updated_at: datetime


class ToolRow(TypedDict):
    id: UUID
    org_id: UUID
    name: str
    description: str
    url: str
    method: str
    headers: dict[str, str]      # JSONB
    request_body: dict[str, Any] # JSONB
    created_at: datetime
    updated_at: datetime


class KnowledgebaseRow(TypedDict):
    id: UUID
    org_id: UUID
    name: str
    vector_index_id: str | None
    document_count: int
    created_at: datetime
    updated_at: datetime


class ContactListRow(TypedDict):
    id: UUID
    org_id: UUID
    name: str
    member_count: int
    created_at: datetime
    updated_at: datetime


class ContactMemberRow(TypedDict):
    id: UUID
    list_id: UUID
    phone_number: str
    variables: dict[str, Any]  # JSONB — per-contact template vars
    created_at: datetime


class PhoneNumberRow(TypedDict):
    id: UUID
    org_id: UUID
    phone_number: str
    friendly_name: str
    country_code: str
    twilio_sid: str
    regulatory_bundle_sid: str | None
    created_at: datetime
