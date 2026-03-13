import json
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from app.dependencies import DBDep, OrgDep, RedisDep
from bolna.models import APIParams, ToolDescription, ToolModel
from db.queries.tools import (
    create_tool as db_create_tool,
)
from db.queries.tools import (
    delete_tool as db_delete_tool,
)
from db.queries.tools import (
    get_tool as db_get_tool,
)
from db.queries.tools import (
    list_tools as db_list_tools,
)
from db.queries.tools import (
    update_test_status as db_update_test_status,
)
from db.queries.tools import (
    update_tool as db_update_tool,
)
from services.tool_service import invalidate_tool_cache

router = APIRouter(prefix="/tools", tags=["tools"])


# ─────────────────────────────── helpers ────────────────────────────────────

def _row_to_dict(row) -> dict:
    """Convert a raw asyncpg tool row to the standard API response dict."""
    raw_schema = row["tool_schema"]
    tool_schema_parsed = json.loads(raw_schema) if isinstance(raw_schema, str) else raw_schema
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "description": row["description"],
        "category": row["category"],
        "tool_schema": tool_schema_parsed,
        "tags": row["tags"] or [],
        "isActive": row["is_active"],
        "lastTested": row["last_tested"].isoformat() if row["last_tested"] else None,
        "testStatus": row["test_status"],
        "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
        "updatedAt": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


def _extract_identity(tool_model: ToolModel):
    """Return (tool_name, tool_description) from the first item in tool_model.tools."""
    item = tool_model.tools[0]
    if isinstance(item, ToolDescription):
        return item.function.name, item.function.description
    # ToolDescriptionLegacy — name/description sit directly on the object
    return item.name, item.description


# ─────────────────────────────── request bodies ─────────────────────────────

class CreateToolBody(BaseModel):
    tools: List[Any]
    tools_params: Dict[str, Any]
    category: str = "custom"
    tags: List[str] = []


class UpdateToolBody(BaseModel):
    tools: List[Any]
    tools_params: Dict[str, Any]
    category: Optional[str] = None
    tags: Optional[List[str]] = None


# ─────────────────────────────── routes ─────────────────────────────────────

@router.post("")
async def create_tool(body: CreateToolBody, db: DBDep, org: OrgDep, redis: RedisDep):
    """Create a new API tool scoped to the caller's organization."""
    try:
        tool = ToolModel(tools=body.tools, tools_params=body.tools_params)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid tool format: {exc}")

    if not tool.tools or len(tool.tools) != 1:
        raise HTTPException(status_code=400, detail="Exactly one tool must be provided")
    if not tool.tools_params or len(tool.tools_params) != 1:
        raise HTTPException(status_code=400, detail="Exactly one tools_params entry must be provided")

    tool_name, tool_description = _extract_identity(tool)
    param_key = next(iter(tool.tools_params))
    if param_key != tool_name:
        raise HTTPException(
            status_code=400,
            detail=f"tools_params key '{param_key}' must match tool name '{tool_name}'",
        )

    row = await db_create_tool(
        db,
        org_id=str(org),
        name=tool_name,
        description=tool_description,
        tool_schema=tool.model_dump(),
        category=body.category,
        tags=body.tags,
    )

    await invalidate_tool_cache(redis, str(org))

    return {
        "success": True,
        "message": f"Tool '{tool_name}' created successfully",
        "data": _row_to_dict(row),
    }


@router.get("")
async def get_tools(
    db: DBDep,
    org: OrgDep,
    category: Optional[str] = Query(default=None),
    is_active: Optional[bool] = Query(default=None),
    search: Optional[str] = Query(default=None),
):
    """List all tools for the caller's organization with optional filters."""
    rows = await db_list_tools(
        db,
        org_id=str(org),
        category=category,
        is_active=is_active,
        search=search,
    )
    tools = [_row_to_dict(r) for r in rows]

    return {
        "success": True,
        "message": f"Found {len(tools)} tools",
        "data": tools,
    }


@router.get("/{tool_id}")
async def get_tool(tool_id: str, db: DBDep, org: OrgDep):
    """Retrieve a single tool by ID."""
    row = await db_get_tool(db, tool_id, str(org))
    if not row:
        raise HTTPException(status_code=404, detail="Tool not found")

    return {
        "success": True,
        "message": "Tool retrieved successfully",
        "data": _row_to_dict(row),
    }


@router.put("/{tool_id}")
async def update_tool(tool_id: str, body: UpdateToolBody, db: DBDep, org: OrgDep, redis: RedisDep):
    """Replace a tool's definition and/or metadata."""
    try:
        tool = ToolModel(tools=body.tools, tools_params=body.tools_params)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid tool format: {exc}")

    if not tool.tools or len(tool.tools) != 1:
        raise HTTPException(status_code=400, detail="Exactly one tool must be provided")
    if not tool.tools_params or len(tool.tools_params) != 1:
        raise HTTPException(status_code=400, detail="Exactly one tools_params entry must be provided")

    tool_name, tool_description = _extract_identity(tool)
    param_key = next(iter(tool.tools_params))
    if param_key != tool_name:
        raise HTTPException(
            status_code=400,
            detail=f"tools_params key '{param_key}' must match tool name '{tool_name}'",
        )

    existing = await db_get_tool(db, tool_id, str(org))
    if not existing:
        raise HTTPException(status_code=404, detail="Tool not found")

    # Fall back to current DB values for optional fields omitted from the request
    final_category = body.category if body.category is not None else existing["category"]
    final_tags = body.tags if body.tags is not None else (existing["tags"] or [])

    updated = await db_update_tool(
        db,
        tool_id=tool_id,
        org_id=str(org),
        name=tool_name,
        description=tool_description,
        tool_schema=tool.model_dump(),
        category=final_category,
        tags=final_tags,
    )

    await invalidate_tool_cache(redis, str(org))

    return {
        "success": True,
        "message": f"Tool '{tool_name}' updated successfully",
        "data": _row_to_dict(updated),
    }


@router.delete("/{tool_id}")
async def delete_tool(tool_id: str, db: DBDep, org: OrgDep, redis: RedisDep):
    """Permanently delete a tool."""
    existing = await db_get_tool(db, tool_id, str(org))
    if not existing:
        raise HTTPException(status_code=404, detail="Tool not found")

    await db_delete_tool(db, tool_id, str(org))

    await invalidate_tool_cache(redis, str(org))

    return {
        "success": True,
        "message": f"Tool '{existing['name']}' deleted successfully",
        "data": {
            "id": str(existing["id"]),
            "name": existing["name"],
            "description": existing["description"],
        },
    }


@router.post("/{tool_id}/test")
async def test_tool(
    tool_id: str,
    db: DBDep,
    org: OrgDep,
    test_params: Dict[str, Any] = Body(default={}),
):
    """Execute the tool's API endpoint with the supplied test parameters."""
    row = await db_get_tool(db, tool_id, str(org))
    if not row:
        raise HTTPException(status_code=404, detail="Tool not found")
    if not row["is_active"]:
        raise HTTPException(status_code=400, detail="Tool is not active")

    raw_schema = row["tool_schema"]
    tool_schema = json.loads(raw_schema) if isinstance(raw_schema, str) else raw_schema
    tool = ToolModel(**tool_schema)

    tool_name, _ = _extract_identity(tool)
    api_params: APIParams = tool.tools_params.get(tool_name)
    if not api_params:
        raise HTTPException(
            status_code=400,
            detail=f"No API params found for tool '{tool_name}'",
        )

    # Extract expected / required params from the tool's JSON Schema
    item = tool.tools[0]
    parameters_schema: dict = (
        item.function.parameters if isinstance(item, ToolDescription) else item.parameters
    ) or {}

    expected_params = set(parameters_schema.get("properties", {}).keys())
    required_params = set(parameters_schema.get("required", []))

    missing = required_params - set(test_params.keys())
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required parameters: {', '.join(sorted(missing))}",
        )
    unexpected = set(test_params.keys()) - expected_params
    if unexpected:
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected parameters: {', '.join(sorted(unexpected))}",
        )

    # Build HTTP request
    url = api_params.url
    method = (api_params.method or "POST").upper()
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "Bolna-Tool-Tester/1.0",
    }
    if api_params.api_token:
        headers["Authorization"] = f"Bearer {api_params.api_token}"

    if method == "GET":
        if test_params:
            separator = "&" if "?" in url else "?"
            url += separator + urllib.parse.urlencode(test_params)
        request_body = None
    else:
        request_body = test_params

    tested_at = datetime.now(timezone.utc)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if request_body is not None:
                response = await client.request(method, url, headers=headers, json=request_body)
            else:
                response = await client.request(method, url, headers=headers)

        test_status = "success" if response.status_code < 400 else "error"
        try:
            response_data = response.json()
        except Exception:
            response_data = {"raw": response.text}

    except httpx.TimeoutException:
        await db_update_test_status(db, tool_id, tested_at, "error")
        raise HTTPException(status_code=408, detail="Tool request timed out")

    except httpx.RequestError as exc:
        await db_update_test_status(db, tool_id, tested_at, "error")
        raise HTTPException(status_code=400, detail=f"Request error: {exc}")

    await db_update_test_status(db, tool_id, tested_at, test_status)

    return {
        "success": True,
        "message": f"Tool '{tool_name}' tested successfully",
        "data": {
            "tool_id": tool_id,
            "tool_name": tool_name,
            "test_status": test_status,
            "http_status": response.status_code,
            "response_data": response_data,
            "request_url": url,
            "request_method": method,
            "test_params": test_params,
            "tested_at": tested_at.isoformat(),
        },
    }
