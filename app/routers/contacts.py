import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.dependencies import DBDep, OrgDep
from db.queries.contacts import (
    check_campaign_dependency,
    delete_contacts_by_list,
    find_contact_by_phone,
    get_contact_list,
    insert_contact,
    list_contact_lists,
    list_contacts,
)
from db.queries.contacts import (
    create_contact_list as db_create_contact_list,
)
from db.queries.contacts import (
    delete_contact_list as db_delete_contact_list,
)

router = APIRouter(prefix="/contacts", tags=["contacts"])


# ─────────────────────────────── request bodies ──────────────────────────────

class ContactModel(BaseModel):
    phone_number: str
    template_variables: Dict[str, Any] = {}


class CreateContactListPayload(BaseModel):
    name: str
    description: str = ""
    contacts: List[ContactModel] = []


class EditContactListPayload(BaseModel):
    contacts: List[ContactModel]


class AddContactsPayload(BaseModel):
    contacts: List[ContactModel]


# ─────────────────────────────── routes ──────────────────────────────────────

@router.post("")
async def create_contact_list(
    payload: CreateContactListPayload, db: DBDep, org: OrgDep
):
    """Create a new contact list and optionally seed it with contacts."""
    try:
        async with db.transaction():
            list_id = await db_create_contact_list(
                db, org, payload.name, payload.description,
            )
            for contact in payload.contacts:
                await insert_contact(
                    db, list_id, org,
                    contact.phone_number, contact.template_variables,
                )
        return {
            "list_id": str(list_id),
            "status": "created",
            "contact_count": len(payload.contacts),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to create contact list") from exc


@router.get("/lists")
async def get_contact_lists(db: DBDep, org: OrgDep):
    """Return all contact lists for the organisation with their contact counts."""
    lists = await list_contact_lists(db, org)
    return {
        "contact_lists": [
            {
                "list_id": str(r["list_id"]),
                "name": r["name"],
                "description": r["description"],
                "created_at": r["created_at"].isoformat(),
                "contact_count": int(r["contact_count"]),
            }
            for r in lists
        ]
    }


@router.get("")
async def get_contacts(
    db: DBDep,
    org: OrgDep,
    list_id: uuid.UUID = Query(...),
):
    """Return all contacts in a specific list."""
    contacts = await list_contacts(db, list_id, org)
    return {
        "contacts": [
            {
                "contact_id": str(c["contact_id"]),
                "phone_number": c["phone_number"],
                "template_variables": c["template_variables"],
                "created_at": c["created_at"].isoformat(),
            }
            for c in contacts
        ]
    }


@router.put("/{list_id}")
async def update_contacts(
    list_id: uuid.UUID,
    payload: EditContactListPayload,
    db: DBDep,
    org: OrgDep,
):
    """Replace all contacts in a list (full replace semantics)."""
    existing = await get_contact_list(db, list_id, org)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail="Contact list not found or doesn't belong to the account",
        )

    try:
        async with db.transaction():
            await delete_contacts_by_list(db, list_id, org)
            added_contacts = []
            for contact in payload.contacts:
                contact_id = await insert_contact(
                    db, list_id, org,
                    contact.phone_number, contact.template_variables,
                )
                added_contacts.append(str(contact_id))

        return {
            "status": "success",
            "message": "Contact list updated successfully",
            "list_id": str(list_id),
            "total_contacts": len(added_contacts),
            "contact_ids": added_contacts,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to update contact list") from exc


@router.post("/{list_id}/add")
async def add_contacts_to_list(
    list_id: uuid.UUID,
    payload: AddContactsPayload,
    db: DBDep,
    org: OrgDep,
):
    """Append contacts to an existing list, skipping duplicates by phone number."""
    existing_list = await get_contact_list(db, list_id, org)
    if not existing_list:
        raise HTTPException(
            status_code=404,
            detail="Contact list not found or doesn't belong to the account",
        )

    try:
        async with db.transaction():
            added_contacts: list = []
            duplicate_contacts: list = []

            for contact in payload.contacts:
                dup = await find_contact_by_phone(db, list_id, contact.phone_number)
                if dup:
                    duplicate_contacts.append(contact.phone_number)
                    continue

                contact_id = await insert_contact(
                    db, list_id, org,
                    contact.phone_number, contact.template_variables,
                )
                added_contacts.append(
                    {"contact_id": str(contact_id), "phone_number": contact.phone_number}
                )

        return {
            "status": "success",
            "message": f"Added {len(added_contacts)} contacts to the list",
            "list_id": str(list_id),
            "added_contacts": added_contacts,
            "duplicate_contacts": duplicate_contacts,
            "total_added": len(added_contacts),
            "total_duplicates": len(duplicate_contacts),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to add contacts to list") from exc


@router.delete("/{list_id}")
async def delete_contact_list(list_id: uuid.UUID, db: DBDep, org: OrgDep):
    """Delete a contact list and all its contacts.

    Refuses deletion when the list is referenced by any campaign.
    """
    try:
        existing = await get_contact_list(db, list_id, org)
        if not existing:
            raise HTTPException(
                status_code=404,
                detail="Contact list not found or doesn't belong to the account",
            )

        campaign = await check_campaign_dependency(db, list_id)
        if campaign:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete list as it is being used in one or more campaigns",
            )

        async with db.transaction():
            await delete_contacts_by_list(db, list_id, org)
            await db_delete_contact_list(db, list_id, org)

        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to delete contact list") from exc
