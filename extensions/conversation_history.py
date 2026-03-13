"""
extensions/conversation_history.py -- persistent conversation transcript storage.

This module handles *storage* of completed conversation transcripts (S3 and
local filesystem).  It is distinct from ``bolna.helpers.conversation_history``
which manages the in-memory message list during a live call.

After a call ends the transcript is persisted here so it can be retrieved
for analytics, compliance review, or re-training.

Storage backends:
  - **S3** (default for production) -- uses ``aiobotocore`` so all I/O is
    non-blocking.
  - **Local** -- writes to ``{PREPROCESS_DIR}/{call_sid}/transcript.json``
    for development convenience.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from aiobotocore.session import AioSession
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# Environment config (same env vars used by bolna.helpers.utils)
BUCKET_NAME = os.getenv("BUCKET_NAME", "kallabot-recordings")
PREPROCESS_DIR = os.getenv("PREPROCESS_DIR", os.path.join(os.getcwd(), "preprocess"))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

async def store_conversation(
    call_sid: str,
    transcript: list[dict[str, Any]],
    *,
    bucket_name: str | None = None,
    local: bool = False,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Save a conversation transcript to persistent storage.

    Parameters
    ----------
    call_sid : str
        Unique call identifier used as the storage key prefix.
    transcript : list[dict]
        The ordered list of conversation messages
        (``[{"role": "...", "content": "..."}, ...]``).
    bucket_name : str | None
        Override the default S3 bucket.
    local : bool
        If ``True``, write to the local filesystem instead of S3.
    metadata : dict | None
        Optional metadata dict (agent_id, org_id, timestamps, etc.)
        persisted alongside the transcript.

    Returns
    -------
    str
        The storage key (S3 key or local file path).
    """
    payload = {
        "call_sid": call_sid,
        "transcript": transcript,
    }
    if metadata:
        payload["metadata"] = metadata

    file_key = f"{call_sid}/transcript.json"

    if local:
        return await _store_local(file_key, payload)
    else:
        return await _store_s3(bucket_name or BUCKET_NAME, file_key, payload)


async def _store_s3(bucket: str, key: str, data: dict) -> str:
    """Upload JSON data to S3."""
    session = AioSession()
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(session.create_client("s3"))
        body = json.dumps(data, default=str)
        try:
            await client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
            logger.info("Stored conversation to s3://%s/%s", bucket, key)
        except (BotoCoreError, ClientError) as exc:
            logger.error("S3 put_object failed for %s: %s", key, exc)
            raise
    return f"s3://{bucket}/{key}"


async def _store_local(key: str, data: dict) -> str:
    """Write JSON data to the local filesystem."""
    full_path = os.path.join(PREPROCESS_DIR, key)
    directory = os.path.dirname(full_path)
    os.makedirs(directory, exist_ok=True)
    try:
        with open(full_path, "w") as f:
            json.dump(data, f, default=str, indent=2)
        logger.info("Stored conversation locally at %s", full_path)
    except OSError as exc:
        logger.error("Failed to write conversation to %s: %s", full_path, exc)
        raise
    return full_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

async def load_conversation(
    call_sid: str,
    *,
    bucket_name: str | None = None,
    local: bool = False,
) -> dict[str, Any] | None:
    """Load a conversation transcript from storage.

    Parameters
    ----------
    call_sid : str
        The call identifier used when storing.
    bucket_name : str | None
        Override the default S3 bucket.
    local : bool
        If ``True``, read from the local filesystem.

    Returns
    -------
    dict | None
        The parsed JSON payload (including ``transcript`` key),
        or ``None`` if the file does not exist or is unreadable.
    """
    file_key = f"{call_sid}/transcript.json"

    if local:
        return await _load_local(file_key)
    else:
        return await _load_s3(bucket_name or BUCKET_NAME, file_key)


async def _load_s3(bucket: str, key: str) -> dict | None:
    """Download and parse JSON from S3."""
    session = AioSession()
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(session.create_client("s3"))
        try:
            response = await client.get_object(Bucket=bucket, Key=key)
            body = await response["Body"].read()
            return json.loads(body)
        except client.exceptions.NoSuchKey:
            logger.warning("Conversation not found in S3: s3://%s/%s", bucket, key)
            return None
        except (BotoCoreError, ClientError) as exc:
            logger.error("S3 get_object failed for %s: %s", key, exc)
            return None
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("Corrupt conversation JSON at s3://%s/%s: %s", bucket, key, exc)
            return None


async def _load_local(key: str) -> dict | None:
    """Read and parse JSON from the local filesystem."""
    full_path = os.path.join(PREPROCESS_DIR, key)
    if not os.path.isfile(full_path):
        logger.warning("Conversation file not found: %s", full_path)
        return None
    try:
        with open(full_path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to read conversation from %s: %s", full_path, exc)
        return None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

async def delete_conversation(
    call_sid: str,
    *,
    bucket_name: str | None = None,
    local: bool = False,
) -> bool:
    """Remove a stored conversation transcript.

    Parameters
    ----------
    call_sid : str
        The call identifier used when storing.
    bucket_name : str | None
        Override the default S3 bucket.
    local : bool
        If ``True``, delete from the local filesystem.

    Returns
    -------
    bool
        ``True`` if the object was deleted (or did not exist), ``False`` on error.
    """
    file_key = f"{call_sid}/transcript.json"

    if local:
        return await _delete_local(file_key)
    else:
        return await _delete_s3(bucket_name or BUCKET_NAME, file_key)


async def _delete_s3(bucket: str, key: str) -> bool:
    """Delete an object from S3."""
    session = AioSession()
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(session.create_client("s3"))
        try:
            await client.delete_object(Bucket=bucket, Key=key)
            logger.info("Deleted conversation from s3://%s/%s", bucket, key)
            return True
        except (BotoCoreError, ClientError) as exc:
            logger.error("S3 delete_object failed for %s: %s", key, exc)
            return False


async def _delete_local(key: str) -> bool:
    """Delete a file from the local filesystem."""
    full_path = os.path.join(PREPROCESS_DIR, key)
    try:
        if os.path.isfile(full_path):
            os.remove(full_path)
            logger.info("Deleted conversation file: %s", full_path)
        # Also try to clean up the parent directory if empty
        parent = os.path.dirname(full_path)
        if os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)
        return True
    except OSError as exc:
        logger.error("Failed to delete conversation at %s: %s", full_path, exc)
        return False
