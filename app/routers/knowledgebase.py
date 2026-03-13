"""
Knowledgebase routes.

main_server.py reference: lines 2445-2650.

Routes (exact paths from main_server.py):
  POST   /knowledgebase           -- upload a document and create a KB record
  GET    /knowledgebases          -- list all KBs for the account (note: plural path)
  DELETE /knowledgebase/{kb_id}   -- remove KB record + local vector store files

NOTE: prefix="" is intentional -- POST and DELETE use /knowledgebase (singular)
while GET uses /knowledgebases (plural). Setting prefix="" and writing full paths
keeps both correct.
"""

from __future__ import annotations

import mimetypes
import os
import shutil
import tempfile
import threading
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.dependencies import DBDep, OrgDep
from db.queries.knowledgebase import (
    create_knowledgebase as db_create_knowledgebase,
)
from db.queries.knowledgebase import (
    delete_knowledgebase as db_delete_knowledgebase,
)
from db.queries.knowledgebase import (
    get_knowledgebase as db_get_knowledgebase,
)
from db.queries.knowledgebase import (
    get_knowledgebase_by_name as db_get_knowledgebase_by_name,
)
from db.queries.knowledgebase import (
    list_knowledgebases as db_list_knowledgebases,
)

# create_table processes the uploaded document into a local LanceDB/RAG vector
# store (see backend/bolna/helpers/data_ingestion_pipe.py). Import it when
# available; fall back gracefully so the router loads in environments that
# don't have the full backend dependency set installed.
try:
    from bolna.helpers.data_ingestion_pipe import create_table  # type: ignore[import]
except ImportError:  # pragma: no cover
    def create_table(table_name: str, file_name: str) -> None:  # type: ignore[misc]
        """Placeholder -- vector ingestion not available in this environment."""
        pass

# No router prefix -- path differences (singular vs. plural) require full paths.
router = APIRouter(tags=["knowledgebase"])

_SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
    "application/msword",  # doc
    "application/rtf",
    "text/plain",
    "text/markdown",
    "text/html",
    "text/xml",
    "application/xml",
    "application/epub+zip",
    "text/csv",
}

_ALLOWED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".txt", ".md",
    ".html", ".htm", ".xml", ".rtf", ".epub", ".csv",
}


# ---------------------------------------------------------------------------
# POST /knowledgebase  -- create a new knowledge base by uploading a document
# ---------------------------------------------------------------------------

@router.post("/knowledgebase")
async def create_knowledgebase(
    org: OrgDep,
    db: DBDep,
    friendly_name: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """Upload a document (PDF, Word, TXT, etc.) and create a knowledge base entry.

    The file is saved under /tmp/, a DB record is inserted, and vector ingestion
    is launched in a background thread (non-blocking).
    """
    # -- file-type validation -------------------------------------------------
    file_extension = (
        os.path.splitext(file.filename.lower())[1] if file.filename else ""
    )
    if (
        file.content_type not in _SUPPORTED_CONTENT_TYPES
        and file_extension not in _ALLOWED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type not supported. Supported formats: PDF, Word (doc/docx), "
                f"text (txt), Markdown (md), HTML, XML, RTF, EPUB, CSV. "
                f"Uploaded: {file.content_type} ({file_extension})"
            ),
        )

    # -- duplicate friendly_name check ----------------------------------------
    existing = await db_get_knowledgebase_by_name(db, org, friendly_name)
    if existing:
        raise HTTPException(status_code=400, detail="Friendly name already exists")

    # -- read & persist file to /tmp/ -----------------------------------------
    file_content = await file.read()

    if not file_extension:
        file_extension = mimetypes.guess_extension(file.content_type) or ".bin"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    tmp.write(file_content)
    tmp.close()

    kb_id = uuid.uuid4()
    vector_store_id = str(kb_id)
    file_name = f"/tmp/{kb_id}{file_extension}"
    os.rename(tmp.name, file_name)

    # -- DB insert ------------------------------------------------------------
    await db_create_knowledgebase(
        db,
        kb_id=kb_id,
        org_id=org,
        name=friendly_name,
        vector_store_id=vector_store_id,
        file_name=file.filename,
        file_size=len(file_content),
        content_type=file.content_type,
    )

    # -- background vector ingestion ------------------------------------------
    thread = threading.Thread(
        target=create_table,
        args=(str(kb_id), file_name),
        daemon=True,
    )
    thread.start()

    return {
        "kb_id": str(kb_id),
        "vector_store_id": vector_store_id,
        "status": "processing",
        "friendly_name": friendly_name,
        "file_type": file.content_type,
        "supported_formats": (
            "PDF, Word documents, text files, Markdown, HTML, XML, RTF, EPUB, CSV"
        ),
    }


# ---------------------------------------------------------------------------
# GET /knowledgebases  -- list all knowledge bases for the account
# ---------------------------------------------------------------------------

@router.get("/knowledgebases")
async def get_knowledgebases(org: OrgDep, db: DBDep) -> dict:
    """Return all knowledge base records owned by the current account."""
    kbs = await db_list_knowledgebases(db, org)

    return {
        "knowledgebases": [
            {
                "kb_id": str(kb["id"]),
                "friendly_name": kb["friendly_name"],
                "vector_store_id": kb["vector_store_id"],
                "status": kb["status"],
                "created_at": kb["created_at"].isoformat(),
            }
            for kb in kbs
        ]
    }


# ---------------------------------------------------------------------------
# DELETE /knowledgebase/{kb_id}  -- delete a knowledge base
# ---------------------------------------------------------------------------

@router.delete("/knowledgebase/{kb_id}")
async def delete_knowledgebase(
    kb_id: uuid.UUID,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Delete a knowledge base record and its local vector store files."""
    kb = await db_get_knowledgebase(db, kb_id, org)
    if not kb:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found or doesn't belong to the account",
        )

    # -- clean up local LanceDB vector store ----------------------------------
    vector_store_path = os.path.join(
        "local_setup", "RAG", f"{kb['vector_store_id']}.lance"
    )
    if os.path.exists(vector_store_path):
        try:
            shutil.rmtree(vector_store_path)
        except OSError as exc:
            # Log but don't block the delete -- the DB row removal is more
            # important than leftover files.
            import logging
            logging.getLogger(__name__).warning(
                "Could not remove vector store files at %s: %s",
                vector_store_path,
                exc,
            )

    # -- DB delete ------------------------------------------------------------
    await db_delete_knowledgebase(db, kb_id, org)

    return {
        "status": "success",
        "message": "Knowledge base deleted successfully",
        "kb_id": str(kb_id),
    }
