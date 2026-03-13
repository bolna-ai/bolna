"""
WebSocket routes -- real-time audio bridge between Twilio/browser and the
Bolna agent pipeline.

Ported from backend/local_setup/main_server.py (lines ~1551-1706, ~2114-2354).

Routes:
  WS  /v1/chat/{agent_id}         -- Twilio Media Streams WebSocket (telephony)
  WS  /web-call/v1/{agent_id}     -- Browser WebRTC client WebSocket (web calls)

Both endpoints:
  1. Accept the WebSocket connection
  2. Authenticate the caller (API key, session token, or Twilio implicit trust)
  3. Load the agent config (temporary agents from Redis, persistent from DB)
  4. Create an AssistantManager instance with the agent config
  5. Run the bidirectional audio loop via assistant_manager.run()
  6. Handle disconnection / cleanup (Redis session keys, call tracking)
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.config import get_settings
from app.dependencies import get_db, get_redis
from bolna.agent_manager.assistant_manager import AssistantManager
from db.queries.agents import get_agent
from db.queries.calls import (
    update_call_status,
    update_call_transcript,
)
from db.queries.organizations import get_org_by_api_key

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["websockets"])

# Plans that are allowed to make calls (matches old server ALLOWED_PLANS)
ALLOWED_PLANS = {"free", "basic", "pro", "business", "enterprise"}

# Track active websocket connections for graceful shutdown
active_websockets: list[WebSocket] = []


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

async def _get_account_from_auth_token(redis, db, auth_token: str) -> Optional[uuid.UUID]:
    """
    Extract account_id from an auth token.  Supports:
      - session tokens (``session_...``)  -- looked up in Redis
      - API keys (``sk_...``)             -- looked up in the DB
    """
    try:
        if auth_token.startswith("session_"):
            # Check web-call session tokens
            account_id_str = await redis.get(f"web_call_session_token:{auth_token}")
            if account_id_str:
                return uuid.UUID(account_id_str)

            # Check chat session tokens
            account_id_str = await redis.get(f"chat_session_token:{auth_token}")
            if account_id_str:
                return uuid.UUID(account_id_str)

        if auth_token.startswith("sk_"):
            row = await get_org_by_api_key(db, auth_token)
            if row:
                return row["account_id"]

        logger.warning("Invalid auth token format: %s...", auth_token[:20])
        return None

    except Exception as e:
        logger.error("Auth token validation error: %s", e)
        return None


async def _check_org_plan_and_minutes(db, account_id: uuid.UUID) -> Optional[str]:
    """
    Verify the organisation is on an allowed plan and has available minutes.

    Returns None on success, or an error message string if the check fails.
    """
    org = await db.fetchrow(
        'SELECT "planType", minutes FROM "Organization" WHERE "accountId" = $1',
        str(account_id),
    )
    if not org:
        return "Organization not found for this account. Cannot make calls."

    raw_plan = org["planType"] or ""
    plan = raw_plan.strip().lower()

    if plan not in ALLOWED_PLANS:
        return f"Your plan ('{raw_plan}') does not allow making calls."

    if org["minutes"] is None or org["minutes"] <= 0:
        return "You do not have enough minutes to make calls. Please upgrade or purchase more minutes."

    return None


async def _load_temp_agent_config(redis, agent_id: str) -> tuple[dict | None, dict | None]:
    """
    Load a temporary agent's config + prompts from Redis.

    Returns (agent_config, agent_prompts) or (None, None) if not found.
    """
    redis_key = f"temp_agent_config:{agent_id}"
    agent_config_json = await redis.get(redis_key)
    if not agent_config_json:
        return None, None

    complete_config = json.loads(agent_config_json)

    if "agent_config" in complete_config:
        agent_config = complete_config["agent_config"]
        agent_prompts = complete_config.get("agent_prompts", {})
    else:
        agent_config = complete_config
        agent_prompts = {}

    # Merge template variables if stored separately
    vars_json = await redis.get(f"template_variables:{agent_id}")
    if vars_json:
        agent_config["template_variables"] = json.loads(vars_json)

    return agent_config, agent_prompts


# ---------------------------------------------------------------------------
# WS  /v1/chat/{agent_id}  --  Twilio Media Streams
# ---------------------------------------------------------------------------

@router.websocket("/v1/chat/{agent_id}")
async def twilio_media_stream(
    websocket: WebSocket,
    agent_id: str,
    auth_token: Optional[str] = Query(None),
    chat_session_id: Optional[str] = Query(None),
    user_agent: Optional[str] = Query(None),
    enable_realtime_events: str = Query("true"),
    event_types: str = Query(
        "transcription_events,synthesis_events,llm_events,pipeline_status"
    ),
) -> None:
    """Twilio Media Streams WebSocket.

    Receives mu-law audio chunks from Twilio, pipes them through the Bolna
    agent pipeline (ASR -> LLM -> TTS), and returns synthesised audio back
    as base64-encoded mu-law.

    Authentication is optional here because Twilio telephony calls connect
    without a client-side auth token -- the call was already authorised when
    the ``/call`` or ``/inbound_call`` HTTP endpoint initiated it.  When an
    ``auth_token`` *is* supplied (e.g. from a chat widget) the session is
    validated against Redis.
    """
    logger.info("Connected to Twilio WS for agent %s", agent_id)

    # -- Acquire shared resources (not injected via Depends for WS routes) --
    redis = await get_redis()
    db_gen = get_db()
    db = await db_gen.__anext__()

    user_account_id: Optional[uuid.UUID] = None

    try:
        # -- Authentication (optional for telephony) -----------------------
        if auth_token:
            user_account_id = await _get_account_from_auth_token(redis, db, auth_token)
            if not user_account_id:
                await websocket.close(code=1008, reason="Invalid authentication token")
                return

            # Session-based auth validation
            if chat_session_id and auth_token.startswith("session_"):
                chat_session_json = await redis.get(f"chat_session:{chat_session_id}")
                if not chat_session_json:
                    await websocket.close(code=1008, reason="Chat session not found or expired")
                    return

                chat_session_data = json.loads(chat_session_json)
                session_account_id = uuid.UUID(chat_session_data["account_id"])

                if session_account_id != user_account_id:
                    await websocket.close(code=1008, reason="Access denied for this chat session")
                    return

                if chat_session_data["agent_id"] != agent_id:
                    await websocket.close(code=1008, reason="Agent ID mismatch for this chat session")
                    return
        else:
            logger.info(
                "No auth_token provided -- assuming Twilio telephony call for agent %s",
                agent_id,
            )

        # -- Accept the WebSocket ------------------------------------------
        await websocket.accept()
        active_websockets.append(websocket)

        # -- Load agent configuration -------------------------------------
        agent_config: dict | None = None
        agent_prompts: dict | None = None

        is_temp_agent = await redis.get(f"is_temp_agent:{agent_id}")

        if is_temp_agent:
            agent_config, agent_prompts = await _load_temp_agent_config(redis, agent_id)
            if not agent_config:
                logger.error("Temporary agent config not found for %s", agent_id)
                await websocket.close(code=1008, reason="Temporary agent configuration not found")
                return
            logger.info("Loaded temporary agent config for %s", agent_id)
        else:
            # Persistent agent -- load from database
            if user_account_id:
                agent_row = await get_agent(db, uuid.UUID(agent_id), user_account_id)
                if not agent_row:
                    await websocket.close(code=1008, reason="Agent not found or access denied")
                    return
            else:
                # Twilio calls without auth_token: load agent without account check.
                # Access was already validated at the call-initiation endpoint.
                agent_row = await db.fetchrow(
                    "SELECT * FROM agents WHERE agent_id = $1 AND deleted_at IS NULL",
                    uuid.UUID(agent_id),
                )
                if not agent_row:
                    await websocket.close(code=1008, reason="Agent not found")
                    return

            agent_config = (
                json.loads(agent_row["agent_config"])
                if isinstance(agent_row["agent_config"], str)
                else agent_row["agent_config"]
            )
            agent_prompts = None
            if agent_row.get("agent_prompts"):
                agent_prompts = (
                    json.loads(agent_row["agent_prompts"])
                    if isinstance(agent_row["agent_prompts"], str)
                    else agent_row["agent_prompts"]
                )

            # Merge template variables if present
            if agent_row.get("template_variables"):
                tv = agent_row["template_variables"]
                agent_config["template_variables"] = (
                    json.loads(tv) if isinstance(tv, str) else tv
                )

        # -- Build context and event config --------------------------------
        context_data = {"recipient_data": {"user_agent": user_agent or "telephony"}}
        parsed_event_types = [e.strip() for e in event_types.split(",") if e.strip()]

        # -- Instantiate the Bolna AssistantManager ------------------------
        assistant_manager = AssistantManager(
            agent_config=agent_config,
            agent_prompts=agent_prompts,
            ws=websocket,
            assistant_id=agent_id,
            context_data=context_data,
            enable_realtime_events=enable_realtime_events.lower() == "true",
            event_types=parsed_event_types,
        )

        # -- Run the bidirectional audio stream ----------------------------
        async for task_id, task_output in assistant_manager.run(local=True):
            logger.info("Twilio WS agent %s task %d output: %s", agent_id, task_id, task_output)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for agent %s", agent_id)
    except Exception as e:
        logger.error("Error in Twilio WS for agent %s: %s", agent_id, e)
        traceback.print_exc()
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1011, reason=str(e))
            except Exception:
                pass
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

        # Clean up session data
        if chat_session_id:
            await redis.delete(f"chat_session:{chat_session_id}")

        if auth_token and auth_token.startswith("session_"):
            await redis.delete(f"chat_session_token:{auth_token}")

        # Release DB connection
        try:
            await db_gen.aclose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WS  /web-call/v1/{agent_id}  --  Browser WebRTC
# ---------------------------------------------------------------------------

@router.websocket("/web-call/v1/{agent_id}")
async def web_call_stream(
    websocket: WebSocket,
    agent_id: str,
    auth_token: str = Query(...),
    call_sid: Optional[str] = Query(None),
    user_agent: Optional[str] = Query(None),
    enforce_streaming: str = Query("true"),
    enable_realtime_events: str = Query("true"),
    event_types: str = Query(
        "transcription_events,synthesis_events,llm_events,pipeline_status"
    ),
) -> None:
    """Browser WebRTC / SDK WebSocket.

    Similar to the Twilio stream but designed for web-based calling:
      - ``auth_token`` is *required* (API key or session token)
      - Receives PCM audio from a browser client (not mu-law)
      - Passes ``is_web_based_call=True`` so the AssistantManager uses the
        correct audio codec and welcome-message logic

    The ``call_sid`` is optional for backward compatibility.  When a session
    token is used (issued by ``POST /web-call``), ``call_sid`` is required to
    prevent duplicate call records.  Legacy API-key callers that omit
    ``call_sid`` will have one auto-generated.
    """
    await websocket.accept()

    # -- Acquire shared resources ------------------------------------------
    redis = await get_redis()
    db_gen = get_db()
    db = await db_gen.__anext__()

    call_start_time = time.time()
    web_call_session_data: dict | None = None
    auto_generated_call_sid: str | None = None
    user_account_id: uuid.UUID | None = None
    tracking_call_sid: str | None = None

    try:
        # -- Authenticate --------------------------------------------------
        user_account_id = await _get_account_from_auth_token(redis, db, auth_token)
        if not user_account_id:
            await websocket.close(code=1008, reason="Authentication failed")
            return

        # Session tokens require call_sid to avoid creating duplicate records
        if auth_token.startswith("session_") and not call_sid:
            logger.warning(
                "Session token used without call_sid; closing to prevent duplicate call records"
            )
            await websocket.close(code=1008, reason="Missing call_sid for session-based web call")
            return

        # -- Validate web-call session (if call_sid provided) --------------
        if call_sid:
            web_call_session_json = await redis.get(f"web_call_session:{call_sid}")
            if not web_call_session_json:
                await websocket.close(code=1008, reason="Web call session not found or expired")
                return

            web_call_session_data = json.loads(web_call_session_json)
            session_account_id = uuid.UUID(web_call_session_data["account_id"])
            session_token = web_call_session_data.get("session_token")

            if session_account_id != user_account_id:
                await websocket.close(code=1008, reason="Access denied for this web call session")
                return

            if auth_token.startswith("session_") and session_token and auth_token != session_token:
                await websocket.close(code=1008, reason="Invalid session token")
                return
        else:
            # Auto-generate a call_sid for tracking when not provided
            auto_generated_call_sid = f"web_call_{uuid.uuid4().hex[:16]}"
            try:
                await db.execute(
                    """
                    INSERT INTO calls (
                        call_sid, agent_id, account_id, call_type, status,
                        from_number, to_number, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    """,
                    auto_generated_call_sid,
                    None,
                    user_account_id,
                    "web_call",
                    "initiated",
                    "web-browser",
                    "web-browser",
                )
                logger.info("Auto-generated web call tracking for %s", auto_generated_call_sid)
            except Exception as exc:
                logger.error("Failed to create auto-generated call record: %s", exc)

        # -- Check organisation plan and minutes ---------------------------
        plan_error = await _check_org_plan_and_minutes(db, user_account_id)
        if plan_error:
            await websocket.close(code=1008, reason=f"Organization access denied: {plan_error}")
            return

        # -- Update call status to connected -------------------------------
        tracking_call_sid = call_sid or auto_generated_call_sid
        if tracking_call_sid:
            try:
                await update_call_status(db, tracking_call_sid, "connected")
            except Exception as exc:
                logger.error("Failed to update web call status to connected: %s", exc)

        # -- Load agent configuration -------------------------------------
        agent_config: dict | None = None
        agent_prompts: dict | None = None

        is_temp_agent = await redis.get(f"is_temp_agent:{agent_id}")

        if is_temp_agent:
            logger.info("Handling temporary agent for web call: %s", agent_id)

            # Security: verify the requester owns this temp agent
            temp_agent_account_str = await redis.get(f"temp_agent_account:{agent_id}")
            if not temp_agent_account_str or uuid.UUID(temp_agent_account_str) != user_account_id:
                logger.warning(
                    "Account mismatch for temp agent %s. User: %s, Creator: %s",
                    agent_id, user_account_id, temp_agent_account_str,
                )
                await websocket.close(code=1008, reason="Authorization failed for temporary agent.")
                return

            agent_config, agent_prompts = await _load_temp_agent_config(redis, agent_id)
            if not agent_config:
                logger.error("Temporary agent config not found for %s", agent_id)
                await websocket.close(code=1008, reason="Temporary agent configuration not found")
                return
        else:
            logger.info("Handling persistent agent for web call: %s", agent_id)
            agent_row = await get_agent(db, uuid.UUID(agent_id), user_account_id)
            if not agent_row:
                await websocket.close(code=1008, reason="Agent not found or access denied")
                return

            agent_config = (
                json.loads(agent_row["agent_config"])
                if isinstance(agent_row["agent_config"], str)
                else agent_row["agent_config"]
            )
            if agent_row.get("agent_prompts"):
                agent_prompts = (
                    json.loads(agent_row["agent_prompts"])
                    if isinstance(agent_row["agent_prompts"], str)
                    else agent_row["agent_prompts"]
                )

        # -- Build context and event config --------------------------------
        context_data = {"recipient_data": {"user_agent": user_agent or "web-browser"}}
        parsed_event_types = [e.strip() for e in event_types.split(",") if e.strip()]

        # -- Instantiate the Bolna AssistantManager ------------------------
        assistant_manager = AssistantManager(
            agent_config=agent_config,
            agent_prompts=agent_prompts,
            ws=websocket,
            assistant_id=agent_id,
            context_data=context_data,
            turn_based_conversation=False,
            is_web_based_call=True,
            enforce_streaming=enforce_streaming.lower() == "true",
            enable_realtime_events=enable_realtime_events.lower() == "true",
            event_types=parsed_event_types,
            call_sid=tracking_call_sid,
        )

        # -- Run the bidirectional audio stream ----------------------------
        async for task_id, task_output in assistant_manager.run(local=False):
            if task_id == 0:
                logger.info("Web call conversation ended for agent %s", agent_id)

    except WebSocketDisconnect as e:
        logger.info("WebSocket disconnected for agent %s: %s - %s", agent_id, e.code, e.reason)
    except Exception as e:
        logger.error("Error in web call for agent %s: %s", agent_id, e, exc_info=True)
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1011, reason=str(e))
            except Exception as close_err:
                logger.error("Error closing websocket: %s", close_err)
    finally:
        # -- Cleanup: track call completion and persist transcript ---------
        tracking_call_sid = call_sid or auto_generated_call_sid
        if tracking_call_sid and user_account_id:
            try:
                call_duration = time.time() - call_start_time

                # Retrieve transcription from Redis (the pipeline stores it there)
                transcription_json_str = await redis.get(f"transcription:{tracking_call_sid}")
                if transcription_json_str:
                    try:
                        transcript_data = json.loads(transcription_json_str)
                        messages = transcript_data if isinstance(transcript_data, list) else transcript_data.get("messages", [])
                        await update_call_transcript(db, tracking_call_sid, messages)
                    except (json.JSONDecodeError, Exception) as exc:
                        logger.error("Failed to persist transcript for %s: %s", tracking_call_sid, exc)

                # Mark call as completed with duration
                try:
                    await db.execute(
                        """UPDATE calls
                           SET status = 'completed', duration = $2, updated_at = NOW()
                           WHERE call_sid = $1""",
                        tracking_call_sid,
                        int(call_duration),
                    )
                except Exception as exc:
                    logger.error("Failed to update web call status to completed: %s", exc)

                # Clean up Redis session data
                if call_sid:
                    await redis.delete(f"web_call_session:{call_sid}")

                if auth_token.startswith("session_"):
                    await redis.delete(f"web_call_session_token:{auth_token}")

                # Clean up transcription key
                await redis.delete(f"transcription:{tracking_call_sid}")

                logger.info(
                    "Web call tracking completed for %s with duration %.2fs",
                    tracking_call_sid, call_duration,
                )

            except Exception as e:
                logger.error("Error in web call cleanup: %s", e)

        # Release DB connection
        try:
            await db_gen.aclose()
        except Exception:
            pass
