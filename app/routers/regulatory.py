"""
Twilio regulatory compliance routes.

All routes proxy to Twilio's Regulatory Compliance API using the org's
subaccount credentials.  30 routes total.

Routes (exact paths from main_server.py):
  GET    /regulatory/requirements
  GET    /regulatory/regulations
  POST   /regulatory/supporting-documents
  GET    /regulatory/supporting-documents
  DELETE /regulatory/supporting-documents/{document_sid}
  GET    /regulatory/supporting-document-types
  POST   /regulatory/end-users
  GET    /regulatory/end-users
  DELETE /regulatory/end-users/{end_user_sid}
  GET    /regulatory/end-user-types
  POST   /regulatory/bundles
  GET    /regulatory/bundles
  GET    /regulatory/bundles/{bundle_sid}
  PATCH  /regulatory/bundles/{bundle_sid}
  DELETE /regulatory/bundles/{bundle_sid}
  POST   /regulatory/bundles/{bundle_sid}/assign
  POST   /regulatory/bundles/{bundle_sid}/evaluations
  GET    /regulatory/bundles/{bundle_sid}/evaluations/{evaluation_sid}
  POST   /regulatory/bundles/{bundle_sid}/copies
  GET    /regulatory/bundles/{bundle_sid}/copies
  POST   /regulatory/bundles/{bundle_sid}/clones
  POST   /regulatory/bundles/{bundle_sid}/replace-items
  GET    /regulatory/bundles/{bundle_sid}/item-assignments
  GET    /regulatory/bundles/{bundle_sid}/item-assignments/{assignment_sid}
  DELETE /regulatory/bundles/{bundle_sid}/item-assignments/{assignment_sid}
  POST   /regulatory/addresses
  GET    /regulatory/addresses
  GET    /regulatory/addresses/{address_sid}
  PATCH  /regulatory/addresses/{address_sid}
  DELETE /regulatory/addresses/{address_sid}
"""

from __future__ import annotations

import base64
import logging
import uuid
from typing import Dict, Optional

import asyncpg
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from twilio.rest import Client

from app.dependencies import DBDep, OrgDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/regulatory", tags=["regulatory"])


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class SupportingDocumentRequest(BaseModel):
    friendly_name: str
    type: str  # e.g. "individual_tax_id", "business_license"
    attributes: Optional[Dict] = None
    file_path: Optional[str] = None
    file_data: Optional[str] = None  # base64-encoded file data
    mime_type: Optional[str] = None


class EndUserRequest(BaseModel):
    friendly_name: str
    type: str  # "individual" or "business"
    attributes: Dict[str, str]


class RegulatoryBundleRequest(BaseModel):
    friendly_name: str
    email: str
    regulation_sid: Optional[str] = None
    iso_country: Optional[str] = None
    end_user_type: Optional[str] = None  # "individual" or "business"
    number_type: Optional[str] = None  # "local", "mobile", "tollfree"


class BundleAssignmentRequest(BaseModel):
    bundle_sid: str
    resource_sid: str  # end-user or supporting-document SID


class AddressRequest(BaseModel):
    friendly_name: str
    customer_name: str
    street: str
    city: str
    region: str
    postal_code: str
    iso_country: str
    street_secondary: Optional[str] = None


class BundleCopyRequest(BaseModel):
    friendly_name: Optional[str] = None


class BundleCloneRequest(BaseModel):
    friendly_name: Optional[str] = None


class BundleReplaceItemsRequest(BaseModel):
    from_bundle_sid: str


# ---------------------------------------------------------------------------
# Helper: build a Twilio Client from the org's subaccount credentials
# ---------------------------------------------------------------------------

async def _twilio_client(db: asyncpg.Connection, org_id: uuid.UUID) -> Client:
    """Return a ``twilio.rest.Client`` configured for the org's subaccount."""
    row = await db.fetchrow(
        "SELECT twilio_subaccount_sid, twilio_subaccount_auth_token "
        "FROM accounts WHERE account_id = $1",
        org_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")
    return Client(row["twilio_subaccount_sid"], row["twilio_subaccount_auth_token"])


# ---------------------------------------------------------------------------
# Requirements & Regulations
# ---------------------------------------------------------------------------

@router.get("/requirements")
async def get_regulatory_requirements(
    org: OrgDep,
    db: DBDep,
    phone_number: str = Query(..., description="Phone number to check requirements for"),
    country: str = Query("US", description="Country code"),
    number_type: str = Query("local", description="Number type (local, mobile, tollfree)"),
) -> dict:
    """Check what regulatory requirements are needed for purchasing a specific phone number."""
    try:
        client = await _twilio_client(db, org)

        # Try the modern approach with includeConstraints first
        regulations_list = []
        try:
            regulations_list = client.numbers.v2.regulatory_compliance.regulations.list(
                iso_country=country.upper(),
                number_type=number_type,
                includeConstraints=True,
                limit=10,
            )
        except Exception:
            regulations_list = client.numbers.v2.regulatory_compliance.regulations.list(
                iso_country=country.upper(),
                number_type=number_type,
                limit=10,
            )

        requirements = []
        for regulation in regulations_list:
            requirement: dict = {
                "regulation_sid": regulation.sid,
                "regulation_name": regulation.friendly_name,
                "iso_country": regulation.iso_country,
                "number_type": regulation.number_type,
                "end_user_type": regulation.end_user_type,
                "requirements": regulation.requirements if hasattr(regulation, "requirements") else {},
                "end_user_types": [],
                "supporting_documents": [],
            }

            # Extract end-user types from the requirements
            if (
                hasattr(regulation, "requirements")
                and regulation.requirements
                and "end_user" in regulation.requirements
            ):
                for end_user_req in regulation.requirements["end_user"]:
                    requirement["end_user_types"].append({
                        "type": end_user_req.get("type"),
                        "friendly_name": end_user_req.get("name"),
                        "requirement_name": end_user_req.get("requirement_name"),
                        "fields": end_user_req.get("fields", []),
                        "detailed_fields": end_user_req.get("detailed_fields", []),
                        "required": True,
                    })

            # Extract supporting-document types from the requirements
            if (
                hasattr(regulation, "requirements")
                and regulation.requirements
                and "supporting_document" in regulation.requirements
            ):
                for doc_group in regulation.requirements["supporting_document"]:
                    for doc_req in doc_group:
                        doc_info: dict = {
                            "type": doc_req.get("type"),
                            "friendly_name": doc_req.get("name"),
                            "requirement_name": doc_req.get("requirement_name"),
                            "description": doc_req.get("description"),
                            "required": True,
                            "accepted_documents": [],
                        }
                        if "accepted_documents" in doc_req:
                            for accepted_doc in doc_req["accepted_documents"]:
                                doc_info["accepted_documents"].append({
                                    "type": accepted_doc.get("type"),
                                    "friendly_name": accepted_doc.get("name"),
                                    "fields": accepted_doc.get("fields", []),
                                    "detailed_fields": accepted_doc.get("detailed_fields", []),
                                })
                        requirement["supporting_documents"].append(doc_info)

            requirements.append(requirement)

        return {
            "phone_number": phone_number,
            "country": country.upper(),
            "number_type": number_type,
            "requirements": requirements,
            "total_regulations": len(requirements),
            "public_documentation": {
                "country_specific_guidelines": f"https://www.twilio.com/en-us/guidelines/{country.lower()}/regulatory",
                "general_regulatory_docs": "https://www.twilio.com/docs/phone-numbers/regulatory",
            },
            "note": (
                "Requirements data may be limited if detailed constraints are not "
                "available in this SDK version. Check public_documentation links "
                "for user-friendly guides."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Regulatory requirements error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get regulatory requirements")


@router.get("/regulations")
async def list_regulations(
    org: OrgDep,
    db: DBDep,
    country: str = Query("US", description="Country code"),
    number_type: str = Query(None, description="Filter by number type"),
    end_user_type: str = Query(None, description="Filter by end user type (individual or business)"),
    include_constraints: bool = Query(False, description="Include detailed constraints and requirements"),
) -> dict:
    """List all available regulations for a country."""
    try:
        client = await _twilio_client(db, org)

        filter_params: dict = {"iso_country": country.upper()}
        if number_type:
            filter_params["number_type"] = number_type
        if end_user_type:
            filter_params["end_user_type"] = end_user_type

        regulations_list = []
        constraints_supported = False
        if include_constraints:
            try:
                filter_params["includeConstraints"] = True
                regulations_list = client.numbers.v2.regulatory_compliance.regulations.list(**filter_params)
                constraints_supported = True
            except Exception:
                filter_params.pop("includeConstraints", None)
                regulations_list = client.numbers.v2.regulatory_compliance.regulations.list(**filter_params)
        else:
            regulations_list = client.numbers.v2.regulatory_compliance.regulations.list(**filter_params)

        regulations_data = []
        for reg in regulations_list:
            regulations_data.append({
                "sid": reg.sid,
                "friendly_name": reg.friendly_name,
                "iso_country": reg.iso_country,
                "number_type": reg.number_type,
                "end_user_type": reg.end_user_type,
                "requirements": (
                    reg.requirements
                    if hasattr(reg, "requirements") and (include_constraints or constraints_supported)
                    else None
                ),
            })

        return {
            "country": country.upper(),
            "number_type_filter": number_type,
            "end_user_type_filter": end_user_type,
            "include_constraints": include_constraints,
            "constraints_supported": constraints_supported,
            "regulations": regulations_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Regulations list error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list regulations")


# ---------------------------------------------------------------------------
# Supporting Documents
# ---------------------------------------------------------------------------

@router.post("/supporting-documents")
async def create_supporting_document(
    request: SupportingDocumentRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Upload a supporting document for regulatory compliance."""
    try:
        client = await _twilio_client(db, org)

        create_params: dict = {
            "friendly_name": request.friendly_name,
            "type": request.type,
        }
        if request.attributes:
            create_params["attributes"] = request.attributes

        def _doc_response(doc, *, note: str | None = None) -> dict:
            resp: dict = {
                "sid": doc.sid,
                "friendly_name": doc.friendly_name,
                "type": doc.type,
                "status": doc.status,
                "account_sid": doc.account_sid,
                "date_created": doc.date_created.isoformat() if doc.date_created else None,
                "date_updated": doc.date_updated.isoformat() if doc.date_updated else None,
            }
            if note:
                resp["note"] = note
            return resp

        if request.file_data and request.mime_type:
            try:
                base64.b64decode(request.file_data)
            except Exception as decode_error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 file data: {decode_error}",
                )
            supporting_document = (
                client.numbers.v2.regulatory_compliance.supporting_documents.create(**create_params)
            )
            return _doc_response(
                supporting_document,
                note=(
                    "Document created successfully. For file uploads, use the Twilio "
                    "Console or upload directly using multipart/form-data to the upload endpoint."
                ),
            )

        if request.file_path:
            supporting_document = (
                client.numbers.v2.regulatory_compliance.supporting_documents.create(**create_params)
            )
            return _doc_response(
                supporting_document,
                note=(
                    "Document created successfully. File upload from server path not "
                    "supported via API. Use Twilio Console for file uploads."
                ),
            )

        supporting_document = (
            client.numbers.v2.regulatory_compliance.supporting_documents.create(**create_params)
        )
        return _doc_response(supporting_document)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Supporting document creation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create supporting document")


@router.get("/supporting-documents")
async def list_supporting_documents(org: OrgDep, db: DBDep) -> dict:
    """List all supporting documents for the account."""
    try:
        client = await _twilio_client(db, org)
        documents = client.numbers.v2.regulatory_compliance.supporting_documents.list()
        return {
            "supporting_documents": [
                {
                    "sid": doc.sid,
                    "friendly_name": doc.friendly_name,
                    "type": doc.type,
                    "status": doc.status,
                    "date_created": doc.date_created.isoformat() if doc.date_created else None,
                    "date_updated": doc.date_updated.isoformat() if doc.date_updated else None,
                }
                for doc in documents
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Supporting documents list: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list supporting documents")


@router.delete("/supporting-documents/{document_sid}")
async def delete_supporting_document(
    document_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Delete a supporting document."""
    try:
        client = await _twilio_client(db, org)
        client.numbers.v2.regulatory_compliance.supporting_documents(document_sid).delete()
        return {"success": True, "message": "Supporting document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting supporting document: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete supporting document: {e}",
        )


@router.get("/supporting-document-types")
async def list_supporting_document_types(org: OrgDep, db: DBDep) -> dict:
    """List all available supporting document types."""
    try:
        client = await _twilio_client(db, org)
        doc_types = client.numbers.v2.regulatory_compliance.supporting_document_types.list()
        return {
            "supporting_document_types": [
                {
                    "sid": getattr(dt, "sid", None),
                    "type": getattr(dt, "type", None),
                    "friendly_name": getattr(dt, "friendly_name", None),
                    "acceptable_extensions": getattr(dt, "acceptable_extensions", None),
                    "description": getattr(dt, "description", None),
                }
                for dt in doc_types
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing supporting document types: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list supporting document types: {e}",
        )


# ---------------------------------------------------------------------------
# End Users
# ---------------------------------------------------------------------------

@router.post("/end-users")
async def create_end_user(
    request: EndUserRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Create an end user for regulatory compliance."""
    try:
        client = await _twilio_client(db, org)
        end_user = client.numbers.v2.regulatory_compliance.end_users.create(
            friendly_name=request.friendly_name,
            type=request.type,
            attributes=request.attributes,
        )
        return {
            "sid": end_user.sid,
            "friendly_name": end_user.friendly_name,
            "type": end_user.type,
            "attributes": end_user.attributes,
            "account_sid": end_user.account_sid,
            "date_created": end_user.date_created.isoformat() if end_user.date_created else None,
            "date_updated": end_user.date_updated.isoformat() if end_user.date_updated else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("End user creation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create end user")


@router.get("/end-users")
async def list_end_users(org: OrgDep, db: DBDep) -> dict:
    """List all end users for the account."""
    try:
        client = await _twilio_client(db, org)
        end_users = client.numbers.v2.regulatory_compliance.end_users.list()
        return {
            "end_users": [
                {
                    "sid": user.sid,
                    "friendly_name": user.friendly_name,
                    "type": user.type,
                    "attributes": user.attributes,
                    "date_created": user.date_created.isoformat() if user.date_created else None,
                    "date_updated": user.date_updated.isoformat() if user.date_updated else None,
                }
                for user in end_users
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("End users list: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list end users")


@router.delete("/end-users/{end_user_sid}")
async def delete_end_user(end_user_sid: str, org: OrgDep, db: DBDep) -> dict:
    """Delete an end user."""
    try:
        client = await _twilio_client(db, org)
        client.numbers.v2.regulatory_compliance.end_users(end_user_sid).delete()
        return {"success": True, "message": "End user deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting end user: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete end user: {e}",
        )


@router.get("/end-user-types")
async def list_end_user_types(org: OrgDep, db: DBDep) -> dict:
    """List all available end user types."""
    try:
        client = await _twilio_client(db, org)
        end_user_types = client.numbers.v2.regulatory_compliance.end_user_types.list()
        return {
            "end_user_types": [
                {
                    "sid": getattr(eut, "sid", None),
                    "type": getattr(eut, "type", None),
                    "friendly_name": getattr(eut, "friendly_name", None),
                    "fields": getattr(eut, "fields", None),
                    "description": getattr(eut, "description", None),
                }
                for eut in end_user_types
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing end user types: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list end user types: {e}",
        )


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------

@router.post("/bundles")
async def create_regulatory_bundle(
    request: RegulatoryBundleRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Create a regulatory bundle for compliance."""
    try:
        if not request.regulation_sid and not (
            request.iso_country and request.end_user_type and request.number_type
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Either 'regulation_sid' OR all of 'iso_country', "
                    "'end_user_type', and 'number_type' must be provided"
                ),
            )

        client = await _twilio_client(db, org)

        create_params: dict = {
            "friendly_name": request.friendly_name,
            "email": request.email,
        }

        if request.regulation_sid:
            create_params["regulation_sid"] = request.regulation_sid
        else:
            create_params["iso_country"] = request.iso_country
            create_params["end_user_type"] = request.end_user_type
            create_params["number_type"] = request.number_type

        bundle = client.numbers.v2.regulatory_compliance.bundles.create(**create_params)

        return {
            "sid": bundle.sid,
            "friendly_name": bundle.friendly_name,
            "status": bundle.status,
            "regulation_sid": (
                bundle.regulation_sid
                if hasattr(bundle, "regulation_sid")
                else request.regulation_sid
            ),
            "account_sid": bundle.account_sid,
            "email": bundle.email if hasattr(bundle, "email") else request.email,
            "date_created": bundle.date_created.isoformat() if bundle.date_created else None,
            "date_updated": bundle.date_updated.isoformat() if bundle.date_updated else None,
            "request_parameters": {
                "iso_country": request.iso_country,
                "end_user_type": request.end_user_type,
                "number_type": request.number_type,
                "regulation_sid": request.regulation_sid,
            },
            "note": (
                "Bundle created in draft status. Use the update endpoint to change "
                "status to pending-review when ready for submission."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Regulatory bundle creation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create regulatory bundle")


@router.get("/bundles")
async def list_regulatory_bundles(
    org: OrgDep,
    db: DBDep,
    status: str = Query(None, description="Filter by status"),
    iso_country: str = Query(None, description="Filter by country"),
    number_type: str = Query(None, description="Filter by number type"),
) -> dict:
    """List all regulatory bundles for the account with enhanced regulation data."""
    try:
        client = await _twilio_client(db, org)

        filter_params: dict = {}
        if status:
            filter_params["status"] = status
        if iso_country:
            filter_params["iso_country"] = iso_country.upper()
        if number_type:
            filter_params["number_type"] = number_type

        bundles = client.numbers.v2.regulatory_compliance.bundles.list(**filter_params)

        enhanced_bundles = []
        for bundle in bundles:
            bundle_iso_country = getattr(bundle, "iso_country", None)
            bundle_end_user_type = getattr(bundle, "end_user_type", None)
            bundle_number_type = getattr(bundle, "number_type", None)
            regulation_sid = getattr(bundle, "regulation_sid", None)

            # If bundle attributes are null, try to extract from regulation
            if regulation_sid and (
                not bundle_iso_country or not bundle_end_user_type or not bundle_number_type
            ):
                try:
                    regulation = client.numbers.v2.regulatory_compliance.regulations(
                        regulation_sid
                    ).fetch()
                    if not bundle_iso_country:
                        bundle_iso_country = getattr(regulation, "iso_country", None)
                    if not bundle_end_user_type:
                        bundle_end_user_type = getattr(regulation, "end_user_type", None)
                    if not bundle_number_type:
                        bundle_number_type = getattr(regulation, "number_type", None)
                except Exception:
                    pass

            enhanced_bundles.append({
                "sid": bundle.sid,
                "friendly_name": getattr(bundle, "friendly_name", None),
                "status": getattr(bundle, "status", None),
                "regulation_sid": regulation_sid,
                "iso_country": bundle_iso_country,
                "end_user_type": bundle_end_user_type,
                "number_type": bundle_number_type,
                "date_created": (
                    bundle.date_created.isoformat()
                    if hasattr(bundle, "date_created") and bundle.date_created
                    else None
                ),
                "date_updated": (
                    bundle.date_updated.isoformat()
                    if hasattr(bundle, "date_updated") and bundle.date_updated
                    else None
                ),
            })

        return {"bundles": enhanced_bundles}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Regulatory bundles list: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list regulatory bundles")


@router.get("/bundles/{bundle_sid}")
async def get_regulatory_bundle(
    bundle_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Get detailed information about a specific regulatory bundle."""
    try:
        client = await _twilio_client(db, org)

        bundle = client.numbers.v2.regulatory_compliance.bundles(bundle_sid).fetch()

        # Get item assignments
        item_assignments = (
            client.numbers.v2.regulatory_compliance.bundles(bundle_sid).item_assignments.list()
        )

        # Get evaluations
        evaluations = (
            client.numbers.v2.regulatory_compliance.bundles(bundle_sid).evaluations.list()
        )

        iso_country = getattr(bundle, "iso_country", None)
        end_user_type = getattr(bundle, "end_user_type", None)
        number_type = getattr(bundle, "number_type", None)
        regulation_sid = getattr(bundle, "regulation_sid", None)

        # If bundle attributes are null, try to extract from regulation
        if regulation_sid and (not iso_country or not end_user_type or not number_type):
            try:
                regulation = client.numbers.v2.regulatory_compliance.regulations(
                    regulation_sid
                ).fetch()
                if not iso_country:
                    iso_country = getattr(regulation, "iso_country", None)
                if not end_user_type:
                    end_user_type = getattr(regulation, "end_user_type", None)
                if not number_type:
                    number_type = getattr(regulation, "number_type", None)
            except Exception:
                pass

        return {
            "bundle": {
                "sid": bundle.sid,
                "friendly_name": getattr(bundle, "friendly_name", None),
                "status": getattr(bundle, "status", None),
                "regulation_sid": regulation_sid,
                "iso_country": iso_country,
                "end_user_type": end_user_type,
                "number_type": number_type,
                "email": getattr(bundle, "email", None),
                "date_created": (
                    bundle.date_created.isoformat()
                    if hasattr(bundle, "date_created") and bundle.date_created
                    else None
                ),
                "date_updated": (
                    bundle.date_updated.isoformat()
                    if hasattr(bundle, "date_updated") and bundle.date_updated
                    else None
                ),
                "valid_until": getattr(bundle, "valid_until", None),
            },
            "item_assignments": [
                {
                    "sid": item.sid,
                    "bundle_sid": getattr(item, "bundle_sid", None),
                    "object_sid": getattr(item, "object_sid", None),
                    "date_created": (
                        item.date_created.isoformat()
                        if hasattr(item, "date_created") and item.date_created
                        else None
                    ),
                }
                for item in item_assignments
            ],
            "evaluations": [
                {
                    "sid": ev.sid,
                    "bundle_sid": getattr(ev, "bundle_sid", None),
                    "status": getattr(ev, "status", None),
                    "results": getattr(ev, "results", None),
                    "date_created": (
                        ev.date_created.isoformat()
                        if hasattr(ev, "date_created") and ev.date_created
                        else None
                    ),
                }
                for ev in evaluations
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting regulatory bundle: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get regulatory bundle: {e}",
        )


@router.patch("/bundles/{bundle_sid}")
async def update_regulatory_bundle(
    bundle_sid: str,
    org: OrgDep,
    db: DBDep,
    status: str = Query(
        ...,
        description=(
            "New status (draft, pending-review, in-review, "
            "twilio-rejected, twilio-approved, provisionally-approved)"
        ),
    ),
) -> dict:
    """Update the status of a regulatory bundle."""
    try:
        client = await _twilio_client(db, org)

        bundle = client.numbers.v2.regulatory_compliance.bundles(bundle_sid).update(
            status=status,
        )

        return {
            "sid": bundle.sid,
            "friendly_name": getattr(bundle, "friendly_name", None),
            "status": getattr(bundle, "status", None),
            "regulation_sid": getattr(bundle, "regulation_sid", None),
            "iso_country": getattr(bundle, "iso_country", None),
            "end_user_type": getattr(bundle, "end_user_type", None),
            "number_type": getattr(bundle, "number_type", None),
            "date_updated": (
                bundle.date_updated.isoformat()
                if hasattr(bundle, "date_updated") and bundle.date_updated
                else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Regulatory bundle update: %s", e)
        raise HTTPException(status_code=500, detail="Failed to update regulatory bundle")


@router.delete("/bundles/{bundle_sid}")
async def delete_regulatory_bundle(
    bundle_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Delete a regulatory bundle."""
    try:
        client = await _twilio_client(db, org)
        client.numbers.v2.regulatory_compliance.bundles(bundle_sid).delete()
        return {"success": True, "message": "Regulatory bundle deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Regulatory bundle deletion: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete regulatory bundle")


# ---------------------------------------------------------------------------
# Bundle Assignments
# ---------------------------------------------------------------------------

@router.post("/bundles/{bundle_sid}/assign")
async def assign_item_to_bundle(
    bundle_sid: str,
    request: BundleAssignmentRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Assign an end user or supporting document to a regulatory bundle."""
    try:
        client = await _twilio_client(db, org)

        assignment = (
            client.numbers.v2.regulatory_compliance
            .bundles(bundle_sid)
            .item_assignments.create(object_sid=request.resource_sid)
        )

        return {
            "sid": assignment.sid,
            "bundle_sid": getattr(assignment, "bundle_sid", None),
            "object_sid": getattr(assignment, "object_sid", None),
            "date_created": (
                assignment.date_created.isoformat()
                if hasattr(assignment, "date_created") and assignment.date_created
                else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error assigning item to bundle: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign item to bundle: {e}",
        )


# ---------------------------------------------------------------------------
# Bundle Evaluations
# ---------------------------------------------------------------------------

@router.post("/bundles/{bundle_sid}/evaluations")
async def create_bundle_evaluation(
    bundle_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Create an evaluation for a regulatory bundle to check compliance."""
    try:
        client = await _twilio_client(db, org)

        evaluation = (
            client.numbers.v2.regulatory_compliance
            .bundles(bundle_sid)
            .evaluations.create()
        )

        return {
            "sid": evaluation.sid,
            "bundle_sid": getattr(evaluation, "bundle_sid", None),
            "status": getattr(evaluation, "status", None),
            "results": getattr(evaluation, "results", None),
            "date_created": (
                evaluation.date_created.isoformat()
                if hasattr(evaluation, "date_created") and evaluation.date_created
                else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Bundle evaluation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create bundle evaluation")


@router.get("/bundles/{bundle_sid}/evaluations/{evaluation_sid}")
async def get_evaluation_status(
    bundle_sid: str,
    evaluation_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Get the status of a specific evaluation."""
    try:
        client = await _twilio_client(db, org)

        evaluation = (
            client.numbers.v2.regulatory_compliance
            .bundles(bundle_sid)
            .evaluations(evaluation_sid)
            .fetch()
        )

        return {
            "sid": evaluation.sid,
            "bundle_sid": getattr(evaluation, "bundle_sid", None),
            "status": getattr(evaluation, "status", None),
            "results": getattr(evaluation, "results", None),
            "date_created": (
                evaluation.date_created.isoformat()
                if hasattr(evaluation, "date_created") and evaluation.date_created
                else None
            ),
            "date_updated": (
                evaluation.date_updated.isoformat()
                if hasattr(evaluation, "date_updated") and evaluation.date_updated
                else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting evaluation status: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get evaluation status: {e}",
        )


# ---------------------------------------------------------------------------
# Bundle Copies
# ---------------------------------------------------------------------------

@router.post("/bundles/{bundle_sid}/copies")
async def create_bundle_copy(
    bundle_sid: str,
    request: BundleCopyRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Create a copy of a bundle for compliance information updates."""
    try:
        client = await _twilio_client(db, org)

        create_params: dict = {}
        if request.friendly_name:
            create_params["friendly_name"] = request.friendly_name

        bundle_copy = (
            client.numbers.v2.regulatory_compliance
            .bundles(bundle_sid)
            .bundle_copies.create(**create_params)
        )

        return {
            "sid": bundle_copy.sid,
            "bundle_sid": bundle_copy.bundle_sid,
            "friendly_name": bundle_copy.friendly_name,
            "status": bundle_copy.status,
            "date_created": (
                bundle_copy.date_created.isoformat() if bundle_copy.date_created else None
            ),
            "date_updated": (
                bundle_copy.date_updated.isoformat() if bundle_copy.date_updated else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating bundle copy: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create bundle copy: {e}",
        )


@router.get("/bundles/{bundle_sid}/copies")
async def list_bundle_copies(
    bundle_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """List all copies of a bundle."""
    try:
        client = await _twilio_client(db, org)

        bundle_copies = (
            client.numbers.v2.regulatory_compliance
            .bundles(bundle_sid)
            .bundle_copies.list()
        )

        return {
            "bundle_copies": [
                {
                    "sid": copy.sid,
                    "bundle_sid": copy.bundle_sid,
                    "friendly_name": copy.friendly_name,
                    "status": copy.status,
                    "date_created": (
                        copy.date_created.isoformat() if copy.date_created else None
                    ),
                    "date_updated": (
                        copy.date_updated.isoformat() if copy.date_updated else None
                    ),
                }
                for copy in bundle_copies
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing bundle copies: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list bundle copies: {e}",
        )


# ---------------------------------------------------------------------------
# Bundle Clones
# ---------------------------------------------------------------------------

@router.post("/bundles/{bundle_sid}/clones")
async def create_bundle_clone(
    bundle_sid: str,
    request: BundleCloneRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Create a clone of a bundle."""
    try:
        client = await _twilio_client(db, org)

        create_params: dict = {}
        if request.friendly_name:
            create_params["friendly_name"] = request.friendly_name

        bundle_clone = (
            client.numbers.v2.regulatory_compliance
            .bundles(bundle_sid)
            .bundle_clones.create(**create_params)
        )

        return {
            "sid": bundle_clone.sid,
            "bundle_sid": bundle_clone.bundle_sid,
            "friendly_name": bundle_clone.friendly_name,
            "status": bundle_clone.status,
            "date_created": (
                bundle_clone.date_created.isoformat() if bundle_clone.date_created else None
            ),
            "date_updated": (
                bundle_clone.date_updated.isoformat() if bundle_clone.date_updated else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating bundle clone: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create bundle clone: {e}",
        )


# ---------------------------------------------------------------------------
# Bundle Replace Items
# ---------------------------------------------------------------------------

@router.post("/bundles/{bundle_sid}/replace-items")
async def replace_bundle_items(
    bundle_sid: str,
    request: BundleReplaceItemsRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Replace items in a bundle with items from another bundle."""
    try:
        client = await _twilio_client(db, org)

        replace_items = (
            client.numbers.v2.regulatory_compliance
            .bundles(bundle_sid)
            .replace_items.create(from_bundle_sid=request.from_bundle_sid)
        )

        return {
            "sid": replace_items.sid,
            "bundle_sid": replace_items.bundle_sid,
            "from_bundle_sid": replace_items.from_bundle_sid,
            "status": replace_items.status,
            "date_created": (
                replace_items.date_created.isoformat() if replace_items.date_created else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error replacing bundle items: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to replace bundle items: {e}",
        )


# ---------------------------------------------------------------------------
# Item Assignments
# ---------------------------------------------------------------------------

@router.get("/bundles/{bundle_sid}/item-assignments")
async def list_item_assignments(
    bundle_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """List all item assignments for a bundle."""
    try:
        client = await _twilio_client(db, org)

        assignments = (
            client.numbers.v2.regulatory_compliance
            .bundles(bundle_sid)
            .item_assignments.list()
        )

        return {
            "item_assignments": [
                {
                    "sid": assignment.sid,
                    "bundle_sid": assignment.bundle_sid,
                    "object_sid": assignment.object_sid,
                    "date_created": (
                        assignment.date_created.isoformat()
                        if assignment.date_created
                        else None
                    ),
                }
                for assignment in assignments
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing item assignments: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list item assignments: {e}",
        )


@router.get("/bundles/{bundle_sid}/item-assignments/{assignment_sid}")
async def get_item_assignment(
    bundle_sid: str,
    assignment_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Get details of a specific item assignment."""
    try:
        client = await _twilio_client(db, org)

        assignment = (
            client.numbers.v2.regulatory_compliance
            .bundles(bundle_sid)
            .item_assignments(assignment_sid)
            .fetch()
        )

        return {
            "sid": assignment.sid,
            "bundle_sid": assignment.bundle_sid,
            "object_sid": assignment.object_sid,
            "date_created": (
                assignment.date_created.isoformat() if assignment.date_created else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting item assignment: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get item assignment: {e}",
        )


@router.delete("/bundles/{bundle_sid}/item-assignments/{assignment_sid}")
async def delete_item_assignment(
    bundle_sid: str,
    assignment_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Remove an item assignment from a bundle."""
    try:
        client = await _twilio_client(db, org)

        client.numbers.v2.regulatory_compliance.bundles(
            bundle_sid
        ).item_assignments(assignment_sid).delete()

        return {"success": True, "message": "Item assignment deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting item assignment: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete item assignment: {e}",
        )


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------

@router.post("/addresses")
async def create_address(
    request: AddressRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Create an address for regulatory compliance."""
    try:
        client = await _twilio_client(db, org)

        address = client.addresses.create(
            friendly_name=request.friendly_name,
            customer_name=request.customer_name,
            street=request.street,
            city=request.city,
            region=request.region,
            postal_code=request.postal_code,
            iso_country=request.iso_country.upper(),
            street_secondary=request.street_secondary,
        )

        return {
            "sid": address.sid,
            "friendly_name": address.friendly_name,
            "customer_name": address.customer_name,
            "street": address.street,
            "street_secondary": address.street_secondary,
            "city": address.city,
            "region": address.region,
            "postal_code": address.postal_code,
            "iso_country": address.iso_country,
            "emergency_enabled": address.emergency_enabled,
            "validated": address.validated,
            "verified": address.verified,
            "date_created": address.date_created.isoformat() if address.date_created else None,
            "date_updated": address.date_updated.isoformat() if address.date_updated else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Address creation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create address")


@router.get("/addresses")
async def list_addresses(
    org: OrgDep,
    db: DBDep,
    customer_name: str = Query(None, description="Filter by customer name"),
    friendly_name: str = Query(None, description="Filter by friendly name"),
    iso_country: str = Query(None, description="Filter by country"),
) -> dict:
    """List all addresses for the account."""
    try:
        client = await _twilio_client(db, org)

        filter_params: dict = {}
        if customer_name:
            filter_params["customer_name"] = customer_name
        if friendly_name:
            filter_params["friendly_name"] = friendly_name
        if iso_country:
            filter_params["iso_country"] = iso_country.upper()

        addresses = client.addresses.list(**filter_params)

        return {
            "addresses": [
                {
                    "sid": addr.sid,
                    "friendly_name": addr.friendly_name,
                    "customer_name": addr.customer_name,
                    "street": addr.street,
                    "street_secondary": addr.street_secondary,
                    "city": addr.city,
                    "region": addr.region,
                    "postal_code": addr.postal_code,
                    "iso_country": addr.iso_country,
                    "emergency_enabled": addr.emergency_enabled,
                    "validated": addr.validated,
                    "verified": addr.verified,
                    "date_created": (
                        addr.date_created.isoformat() if addr.date_created else None
                    ),
                    "date_updated": (
                        addr.date_updated.isoformat() if addr.date_updated else None
                    ),
                }
                for addr in addresses
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing addresses: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list addresses: {e}",
        )


@router.get("/addresses/{address_sid}")
async def get_address(
    address_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Get detailed information about a specific address."""
    try:
        client = await _twilio_client(db, org)

        address = client.addresses(address_sid).fetch()

        return {
            "sid": address.sid,
            "friendly_name": address.friendly_name,
            "customer_name": address.customer_name,
            "street": address.street,
            "street_secondary": address.street_secondary,
            "city": address.city,
            "region": address.region,
            "postal_code": address.postal_code,
            "iso_country": address.iso_country,
            "emergency_enabled": address.emergency_enabled,
            "validated": address.validated,
            "verified": address.verified,
            "date_created": address.date_created.isoformat() if address.date_created else None,
            "date_updated": address.date_updated.isoformat() if address.date_updated else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting address: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get address: {e}",
        )


@router.patch("/addresses/{address_sid}")
async def update_address(
    address_sid: str,
    request: AddressRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Update an existing address."""
    try:
        client = await _twilio_client(db, org)

        address = client.addresses(address_sid).update(
            friendly_name=request.friendly_name,
            customer_name=request.customer_name,
            street=request.street,
            city=request.city,
            region=request.region,
            postal_code=request.postal_code,
            street_secondary=request.street_secondary,
        )

        return {
            "sid": address.sid,
            "friendly_name": address.friendly_name,
            "customer_name": address.customer_name,
            "street": address.street,
            "street_secondary": address.street_secondary,
            "city": address.city,
            "region": address.region,
            "postal_code": address.postal_code,
            "iso_country": address.iso_country,
            "date_updated": address.date_updated.isoformat() if address.date_updated else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating address: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update address: {e}",
        )


@router.delete("/addresses/{address_sid}")
async def delete_address(
    address_sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Delete an address."""
    try:
        client = await _twilio_client(db, org)
        client.addresses(address_sid).delete()
        return {"success": True, "message": "Address deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting address: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete address: {e}",
        )
