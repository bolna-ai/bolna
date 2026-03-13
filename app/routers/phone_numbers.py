"""
Phone number provisioning routes.

Routes (exact paths from old backend main_server.py):
  GET    /available-numbers                -- search available phone numbers
  GET    /account-phone-numbers            -- list org's phone numbers
  GET    /check-free-number-eligibility    -- check if org eligible for free number
  POST   /purchase-number                  -- purchase a phone number
  POST   /purchase-number-with-compliance  -- purchase with compliance docs
  POST   /cancel-number                    -- cancel/release a phone number
  POST   /update-number                    -- update phone number configuration
  POST   /create-emergency-address         -- create emergency address for number
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import stripe
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

from app.config import get_settings
from app.dependencies import DBDep, OrgDep
from db.queries.organizations import (
    get_free_number_eligibility,
    get_org_and_twilio_for_purchase,
    get_twilio_credentials,
    mark_free_phone_number_used,
)
from db.queries.phone_numbers import (
    check_ownership_legacy,
    check_ownership_new,
    create_phone_number,
    delete_legacy_phone_number,
    delete_phone_number_by_sid,
    get_legacy_phone_number_by_sid,
    get_legacy_phone_numbers,
    get_phone_number_by_twilio_sid,
    insert_legacy_phone_number,
    update_legacy_friendly_name,
    update_phone_number_friendly_name,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["phone_numbers"])

# Plans eligible for one free phone number
FREE_NUMBER_ELIGIBLE_PLANS = {"basic", "pro", "business", "enterprise"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_twilio_credentials(db, account_id: uuid.UUID) -> dict:
    """Fetch Twilio sub-account credentials for the given account."""
    row = await get_twilio_credentials(db, account_id)
    if (
        not row
        or not row["twilio_subaccount_sid"]
        or not row["twilio_subaccount_auth_token"]
    ):
        raise HTTPException(
            status_code=400,
            detail="Twilio credentials not configured for this account",
        )
    return row


async def _get_twilio_client(db, account_id: uuid.UUID) -> Client:
    """Return a Twilio ``Client`` using the org's sub-account credentials."""
    creds = await _get_twilio_credentials(db, account_id)
    return Client(
        creds["twilio_subaccount_sid"],
        creds["twilio_subaccount_auth_token"],
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PurchaseNumberRequest(BaseModel):
    phone_number: str
    friendly_name: Optional[str] = None
    payment_method_id: str
    bundle_sid: Optional[str] = None
    bundle_id: Optional[str] = None          # alias for frontend compat
    address_sid: Optional[str] = None
    emergency_address_sid: Optional[str] = None
    voice_url: Optional[str] = None
    sms_url: Optional[str] = None
    country_code: Optional[str] = "US"
    number_type: Optional[str] = "local"
    stripe_customer_id: Optional[str] = None
    is_free_number: Optional[bool] = False
    coupon_code: Optional[str] = None


class UpdateNumberPayload(BaseModel):
    sid: str
    friendly_name: str


class CreateEmergencyAddressRequest(BaseModel):
    customer_name: str
    street: str
    street_secondary: Optional[str] = ""
    city: str
    region: str                               # State / Province code
    postal_code: str
    iso_country: str = "US"
    phone_number: Optional[str] = None        # for logging


# ---------------------------------------------------------------------------
# GET /available-numbers
# ---------------------------------------------------------------------------

@router.get("/available-numbers")
async def get_available_numbers(
    org: OrgDep,
    db: DBDep,
    country: str = Query("US", description="Country code (e.g., US, GB, CA)"),
    number_type: str = Query(
        "local", description="Number type (local, mobile, toll-free, national)"
    ),
    area_code: str = Query(
        None, description="Area code or number pattern to search for"
    ),
    limit: int = Query(20, description="Number of results to return"),
) -> dict:
    """Search for available phone numbers from Twilio."""
    try:
        client = await _get_twilio_client(db, org)
        numbers = client.available_phone_numbers(country)

        kwargs: dict = {"limit": limit}
        if area_code:
            pattern = area_code.strip()
            if not pattern.endswith("*"):
                pattern = pattern + "*"
            kwargs["contains"] = pattern
            logger.info(
                "Searching for numbers with pattern: %s in country: %s",
                pattern, country,
            )

        logger.info(
            "Search parameters: country=%s, number_type=%s, kwargs=%s",
            country, number_type, kwargs,
        )

        if number_type in ("local", "national"):
            available_numbers = numbers.local.list(**kwargs)
        elif number_type == "mobile":
            available_numbers = numbers.mobile.list(**kwargs)
        elif number_type == "toll-free":
            available_numbers = numbers.toll_free.list(**kwargs)
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid number type: {number_type}"
            )

        formatted_numbers = [
            {
                "friendly_name": n.friendly_name,
                "phone_number": n.phone_number,
                "lata": n.lata,
                "locality": n.locality,
                "rate_center": n.rate_center,
                "latitude": n.latitude,
                "longitude": n.longitude,
                "region": n.region,
                "postal_code": n.postal_code,
                "iso_country": n.iso_country,
                "capabilities": {
                    "voice": n.capabilities.get("voice", False),
                    "sms": n.capabilities.get("sms", False),
                    "mms": n.capabilities.get("mms", False),
                },
            }
            for n in available_numbers
        ]

        logger.info(
            "Found %d numbers for country=%s, pattern=%s",
            len(formatted_numbers),
            country,
            kwargs.get("contains", "none"),
        )
        return {"available_phone_numbers": formatted_numbers}

    except TwilioRestException as e:
        logger.error("Twilio API error: %s", e)
        raise HTTPException(
            status_code=e.status, detail=f"Twilio API error: {e}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Internal server error: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}"
        )


# ---------------------------------------------------------------------------
# GET /account-phone-numbers
# ---------------------------------------------------------------------------

@router.get("/account-phone-numbers")
async def get_account_phone_numbers(org: OrgDep, db: DBDep) -> dict:
    """List phone numbers owned by the organisation's Twilio sub-account."""
    try:
        # Fetch numbers directly from Twilio
        client = await _get_twilio_client(db, org)
        incoming_phone_numbers = client.incoming_phone_numbers.list(limit=20)

        # Attempt to load legacy PhoneNumber records for Stripe subscription info
        legacy_by_sid: dict[str, dict] = {}
        try:
            legacy_rows = await get_legacy_phone_numbers(db, org)
            for row in legacy_rows:
                legacy_by_sid[row["sid"]] = dict(row)
        except Exception:
            # Legacy table may not exist in the new schema
            pass

        formatted_numbers = []
        for number in incoming_phone_numbers:
            legacy_rec = legacy_by_sid.get(number.sid)

            cost_info = {"amount": "$2.99", "period": "month", "is_free": False}
            renewal_formatted = "No subscription"

            subscription_id = (
                legacy_rec.get("stripeSubscriptionId") if legacy_rec else None
            )

            if subscription_id:
                try:
                    subscription = stripe.Subscription.retrieve(subscription_id)
                    renewal_date = datetime.fromtimestamp(
                        subscription.current_period_end
                    )
                    renewal_formatted = renewal_date.strftime("%m/%d/%Y")

                    items_data = (
                        subscription.items.data
                        if hasattr(subscription.items, "data")
                        else []
                    )
                    price_amount = (
                        items_data[0].price.unit_amount if items_data else 0
                    )
                    has_discount = (
                        subscription.discount
                        and subscription.discount.coupon.percent_off == 100
                    )

                    if price_amount == 0 or has_discount:
                        cost_info = {
                            "amount": "$0.00",
                            "period": "month",
                            "is_free": True,
                        }
                    else:
                        actual_amount = price_amount / 100
                        cost_info = {
                            "amount": f"${actual_amount:.2f}",
                            "period": "month",
                            "is_free": False,
                        }
                except Exception as e:
                    logger.error("Error fetching subscription details: %s", e)
                    renewal_formatted = "Unknown"

            formatted_numbers.append(
                {
                    "sid": number.sid,
                    "phone_number": number.phone_number,
                    "friendly_name": number.friendly_name,
                    "capabilities": number.capabilities,
                    "status": (
                        "inbound"
                        if number.sms_url or number.voice_url
                        else "outbound"
                    ),
                    "cost": cost_info,
                    "renewal_date": renewal_formatted,
                }
            )

        return {"phone_numbers": formatted_numbers}

    except TwilioRestException as e:
        logger.error("Twilio API error: %s", e)
        raise HTTPException(
            status_code=e.status, detail=f"Twilio API error: {e}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Internal server error: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}"
        )


# ---------------------------------------------------------------------------
# GET /check-free-number-eligibility
# ---------------------------------------------------------------------------

@router.get("/check-free-number-eligibility")
async def check_free_number_eligibility(org: OrgDep, db: DBDep) -> dict:
    """
    Check if the account is eligible for a free phone number.
    This is a secure, server-side only check that cannot be manipulated by
    the frontend.
    """
    try:
        account_info = await get_free_number_eligibility(db, org)

        if not account_info:
            raise HTTPException(status_code=404, detail="Account not found")

        org_plan = account_info["planType"] or "free"
        has_used_free_number = account_info["freePhoneNumberUsed"] or False
        is_paid_plan = org_plan.lower() in FREE_NUMBER_ELIGIBLE_PLANS

        is_eligible = is_paid_plan and not has_used_free_number

        return {
            "isEligible": is_eligible,
            "isPaidPlan": is_paid_plan,
            "hasUsedFreeNumber": has_used_free_number,
            "currentPlan": org_plan,
            "message": (
                "You're eligible for one free phone number!"
                if is_eligible
                else (
                    "Free number already used"
                    if has_used_free_number
                    else (
                        "Upgrade to a paid plan to get a free number "
                        f"(current plan: {org_plan})"
                    )
                )
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error checking free number eligibility: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to check eligibility"
        )


# ---------------------------------------------------------------------------
# POST /create-emergency-address
# ---------------------------------------------------------------------------

@router.post("/create-emergency-address")
async def create_emergency_address(
    request: CreateEmergencyAddressRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """
    Create an emergency address for US phone numbers.
    Required for 911 services to avoid $75 per call charges.
    """
    try:
        client = await _get_twilio_client(db, org)

        if request.iso_country.upper() != "US":
            raise HTTPException(
                status_code=400,
                detail="Emergency addresses are currently only supported "
                       "for US phone numbers",
            )

        logger.info(
            "Creating emergency address for account %s: %s, %s, %s",
            org, request.street, request.city, request.region,
        )

        address = client.addresses.create(
            customer_name=request.customer_name,
            street=request.street,
            street_secondary=request.street_secondary,
            city=request.city,
            region=request.region,
            postal_code=request.postal_code,
            iso_country=request.iso_country,
            emergency_enabled=True,
        )

        logger.info(
            "Successfully created emergency address with SID: %s", address.sid
        )

        return {
            "success": True,
            "address_sid": address.sid,
            "customer_name": address.customer_name,
            "street": address.street,
            "street_secondary": address.street_secondary,
            "city": address.city,
            "region": address.region,
            "postal_code": address.postal_code,
            "iso_country": address.iso_country,
            "emergency_enabled": True,
            "message": "Emergency address created and validated successfully",
        }

    except TwilioRestException as e:
        error_code = getattr(e, "code", None)
        error_message = getattr(e, "msg", str(e))
        logger.error(
            "Twilio error creating emergency address: Code %s, Message: %s",
            error_code, error_message,
        )

        error_map = {
            21401: (
                "Invalid address format. Please check your address details."
            ),
            21402: (
                "Address validation failed. Please ensure your address is "
                "valid and try again."
            ),
            21603: (
                "Address not found in Master Street Address Guide. "
                "Please verify your address is correct."
            ),
            21604: (
                "Invalid ZIP code. Please enter a valid 5-digit ZIP code."
            ),
        }
        detail = error_map.get(error_code)
        if not detail:
            msg_lower = error_message.lower()
            if "address validation" in msg_lower:
                detail = (
                    "Address validation failed. Please ensure your "
                    "address is correct and complete."
                )
            elif "msag" in msg_lower:
                detail = (
                    "Address not found in Master Street Address Guide. "
                    "Please verify your address."
                )
            else:
                detail = (
                    f"Failed to create emergency address: {error_message}"
                )

        raise HTTPException(status_code=400, detail=detail)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating emergency address: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create emergency address: {e}",
        )


# ---------------------------------------------------------------------------
# POST /purchase-number
# ---------------------------------------------------------------------------

@router.post("/purchase-number")
async def purchase_phone_number(
    request: PurchaseNumberRequest,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Purchase a phone number from Twilio and create a billing subscription."""
    subscription_id: str | None = None

    try:
        # -- Validate phone number format --------------------------------
        if not request.phone_number.startswith("+"):
            raise HTTPException(
                status_code=400,
                detail="Phone number must be in E.164 format (start with +)",
            )

        # Handle bundle_id alias for frontend compatibility
        bundle_sid = request.bundle_sid or request.bundle_id
        logger.info(
            "Bundle processing: request.bundle_sid=%s, "
            "request.bundle_id=%s, final bundle_sid=%s",
            request.bundle_sid, request.bundle_id, bundle_sid,
        )

        if bundle_sid and not bundle_sid.startswith("BU"):
            raise HTTPException(
                status_code=400,
                detail="Bundle SID must start with 'BU'",
            )
        if request.address_sid and not request.address_sid.startswith("AD"):
            raise HTTPException(
                status_code=400,
                detail="Address SID must start with 'AD'",
            )

        # -- Get org info + Twilio credentials ---------------------------
        account_info = await get_org_and_twilio_for_purchase(db, org)
        if not account_info:
            raise HTTPException(status_code=404, detail="Account not found")

        # -- Free-number eligibility (server-side only) ------------------
        org_plan = account_info["planType"] or "free"
        has_used_free_number = account_info["freePhoneNumberUsed"] or False
        is_paid_plan = org_plan.lower() in FREE_NUMBER_ELIGIBLE_PLANS
        is_eligible_for_free = is_paid_plan and not has_used_free_number

        logger.info(
            "Free number eligibility for account %s: plan=%s, "
            "used_free=%s, is_paid=%s, eligible=%s",
            org, org_plan, has_used_free_number,
            is_paid_plan, is_eligible_for_free,
        )
        if is_eligible_for_free:
            logger.info(
                "Account %s eligible for free number - applying automatically",
                org,
            )

        # -- Twilio client -----------------------------------------------
        client = Client(
            account_info["twilio_subaccount_sid"],
            account_info["twilio_subaccount_auth_token"],
        )

        # -- Check availability ------------------------------------------
        phone_number_digits = (
            request.phone_number
            .replace("+", "").replace("-", "")
            .replace("(", "").replace(")", "")
            .replace(" ", "")
        )
        country_code = request.country_code or "US"
        number_type = request.number_type or "local"

        try:
            logger.info(
                "Searching for %s numbers in %s containing %s",
                number_type, country_code, phone_number_digits,
            )
            if number_type == "mobile":
                available_numbers = (
                    client.available_phone_numbers(country_code)
                    .mobile.list(contains=phone_number_digits)
                )
            elif number_type == "toll_free":
                available_numbers = (
                    client.available_phone_numbers(country_code)
                    .toll_free.list(contains=phone_number_digits)
                )
            else:  # default to local
                available_numbers = (
                    client.available_phone_numbers(country_code)
                    .local.list(contains=phone_number_digits)
                )
            logger.info(
                "Found %d available %s numbers in %s",
                len(available_numbers), number_type, country_code,
            )

        except TwilioRestException as e:
            logger.error(
                "Error checking number availability for %s %s: %s",
                country_code, number_type, e,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unable to verify phone number availability in "
                    f"{country_code} for {number_type} numbers: {e}"
                ),
            )

        if not available_numbers:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Phone number is not available for purchase as a "
                    f"{number_type} number in {country_code}. "
                    "Please try a different number or contact support."
                ),
            )

        # -- Create Stripe subscription ----------------------------------
        stripe_customer_id = account_info["stripe_customer_id"]
        if not stripe_customer_id:
            raise HTTPException(
                status_code=400, detail="No payment method configured"
            )

        try:
            common_params: dict = dict(
                customer=stripe_customer_id,
                payment_behavior="error_if_incomplete",
                default_payment_method=request.payment_method_id,
                expand=["latest_invoice.payment_intent"],
            )

            if is_eligible_for_free:
                logger.info(
                    "Creating FREE phone number subscription for account %s",
                    org,
                )
                subscription = stripe.Subscription.create(
                    **common_params,
                    items=[{"price": "price_1Ra3A6REurM0XFcMPyAMdofs"}],
                    metadata={
                        "phone_number": request.phone_number,
                        "account_id": str(org),
                        "type": "phone_number_subscription",
                        "is_free_number": "true",
                    },
                )
            else:
                subscription = stripe.Subscription.create(
                    **common_params,
                    items=[
                        {"price": os.getenv("STRIPE_PHONE_NUMBER_PRICE_ID")}
                    ],
                    metadata={
                        "phone_number": request.phone_number,
                        "account_id": str(org),
                        "type": "phone_number_subscription",
                    },
                )

            subscription_id = subscription.id
            logger.info(
                "Subscription created: %s, status: %s",
                subscription_id, subscription.status,
            )

        except stripe.error.CardError as e:
            logger.error(
                "Payment failed for phone number subscription: %s", e
            )
            raise HTTPException(
                status_code=400,
                detail=f"Payment failed: {e.user_message}",
            )
        except stripe.error.InvalidRequestError as e:
            logger.error("Invalid payment request: %s", e)
            raise HTTPException(
                status_code=400, detail=f"Payment setup error: {e}"
            )
        except stripe.error.StripeError as e:
            logger.error("Stripe error creating subscription: %s", e)
            raise HTTPException(
                status_code=400,
                detail=f"Payment processing failed: {e}",
            )

        # -- Purchase number from Twilio ---------------------------------
        try:
            purchase_params: dict = {
                "phone_number": request.phone_number,
                "friendly_name": (
                    request.friendly_name or request.phone_number
                ),
            }

            if bundle_sid:
                purchase_params["bundle_sid"] = bundle_sid
                logger.info(
                    "Using bundle_sid %s for regulatory compliance",
                    bundle_sid,
                )
            if request.address_sid:
                purchase_params["address_sid"] = request.address_sid
                logger.info(
                    "Using address_sid %s for regulatory compliance",
                    request.address_sid,
                )
            if request.emergency_address_sid:
                purchase_params["emergency_address_sid"] = (
                    request.emergency_address_sid
                )
                logger.info(
                    "Using emergency_address_sid %s for US 911 services",
                    request.emergency_address_sid,
                )

            # Voice / SMS URLs
            if request.voice_url:
                purchase_params["voice_url"] = request.voice_url
            else:
                purchase_params["voice_url"] = (
                    f"{settings.base_url}/inbound_call"
                )

            if request.sms_url:
                purchase_params["sms_url"] = request.sms_url

            purchased_number = client.incoming_phone_numbers.create(
                **purchase_params
            )
            logger.info(
                "Successfully purchased phone number %s with SID %s",
                purchased_number.phone_number, purchased_number.sid,
            )

        except TwilioRestException as e:
            error_code = getattr(e, "code", None)
            error_message = getattr(e, "msg", str(e))
            logger.error(
                "Twilio error purchasing %s: Code %s, Message: %s",
                request.phone_number, error_code, error_message,
            )

            _twilio_error_map = {
                21422: "This phone number is not available for purchase",
                21421: "Invalid phone number format",
                21450: (
                    "Regulatory requirements not met. Please ensure you "
                    "have the required bundles and addresses"
                ),
                21619: (
                    "Bundle validation failed. Please check your bundle "
                    "status"
                ),
                21620: (
                    "Address validation failed. Please check your address "
                    "information"
                ),
                21624: (
                    "Bundle is not approved yet. Please wait for bundle "
                    "approval before purchasing"
                ),
                20003: (
                    "Authentication failed. Please check your Twilio "
                    "credentials"
                ),
                20404: "Phone number not found or no longer available",
            }
            detail = _twilio_error_map.get(
                error_code,
                f"Failed to purchase phone number: {error_message}",
            )

            _cancel_subscription_on_failure(subscription_id)
            raise HTTPException(status_code=400, detail=detail)

        except Exception as e:
            logger.error("Unexpected error purchasing phone number: %s", e)
            _cancel_subscription_on_failure(subscription_id)
            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to purchase phone number due to unexpected error"
                ),
            )

        # -- Persist to database -----------------------------------------
        await create_phone_number(
            db,
            org_id=str(org),
            phone_number=purchased_number.phone_number,
            friendly_name=(
                request.friendly_name or purchased_number.friendly_name
            ),
            country_code=country_code,
            twilio_sid=purchased_number.sid,
        )

        # Also write to legacy PhoneNumber table if it exists
        try:
            await insert_legacy_phone_number(
                db,
                phone_number=purchased_number.phone_number,
                friendly_name=request.friendly_name,
                sid=purchased_number.sid,
                organization_id=account_info["organization_id"],
                stripe_subscription_id=subscription_id,
            )
        except Exception:
            # Legacy table may not exist -- not critical
            pass

        # Mark free number as used
        if is_eligible_for_free:
            try:
                await mark_free_phone_number_used(
                    db, account_info["organization_id"],
                )
                logger.info(
                    "Marked free phone number as used for org %s",
                    account_info["organization_id"],
                )
            except Exception as e:
                logger.error("Failed to mark free number as used: %s", e)

        # -- Build response ----------------------------------------------
        capabilities = {}
        if hasattr(purchased_number, "capabilities"):
            capabilities = {
                "voice": getattr(
                    purchased_number.capabilities, "voice", True
                ),
                "sms": getattr(
                    purchased_number.capabilities, "sms", True
                ),
                "mms": getattr(
                    purchased_number.capabilities, "mms", True
                ),
            }
        else:
            capabilities = {"voice": True, "sms": True, "mms": True}

        date_created = getattr(purchased_number, "date_created", None)
        if isinstance(date_created, datetime):
            date_created = date_created.isoformat()

        return {
            "success": True,
            "phone_number": purchased_number.phone_number,
            "sid": purchased_number.sid,
            "friendly_name": purchased_number.friendly_name,
            "subscription_id": subscription_id,
            "is_free_number": is_eligible_for_free,
            "message": (
                "Free phone number allocated!"
                if is_eligible_for_free
                else "Phone number purchased successfully!"
            ),
            "bundle_sid": getattr(
                purchased_number, "bundle_sid", bundle_sid
            ),
            "address_sid": getattr(
                purchased_number, "address_sid", request.address_sid
            ),
            "capabilities": capabilities,
            "address_requirements": getattr(
                purchased_number, "address_requirements", None
            ),
            "date_created": date_created,
            "voice_url": getattr(
                purchased_number, "voice_url", request.voice_url
            ),
            "sms_url": getattr(
                purchased_number, "sms_url", request.sms_url
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in purchase_phone_number: %s", e)
        _cancel_subscription_on_failure(subscription_id)
        raise HTTPException(
            status_code=500,
            detail=(
                "Internal server error occurred while purchasing phone number"
            ),
        )


def _cancel_subscription_on_failure(subscription_id: str | None) -> None:
    """Best-effort cancellation of a Stripe subscription on purchase failure."""
    if not subscription_id:
        return
    try:
        stripe.Subscription.cancel(subscription_id)
        logger.info(
            "Cancelled Stripe subscription %s due to purchase failure",
            subscription_id,
        )
    except Exception as err:
        logger.error(
            "Failed to cancel Stripe subscription %s: %s",
            subscription_id, err,
        )


# ---------------------------------------------------------------------------
# POST /purchase-number-with-compliance
# ---------------------------------------------------------------------------

@router.post("/purchase-number-with-compliance")
async def purchase_phone_number_with_compliance(
    org: OrgDep,
    db: DBDep,
    phone_number: str = Query(
        ..., description="Phone number to purchase"
    ),
    friendly_name: str = Query(
        None, description="Friendly name for the number"
    ),
    bundle_sid: str = Query(
        ..., description="Regulatory bundle SID for compliance"
    ),
) -> dict:
    """Purchase a phone number with regulatory compliance bundle."""
    try:
        creds = await _get_twilio_credentials(db, org)
        client = Client(
            creds["twilio_subaccount_sid"],
            creds["twilio_subaccount_auth_token"],
        )

        # Verify the bundle is approved
        bundle = (
            client.numbers.v2.regulatory_compliance
            .bundles(bundle_sid).fetch()
        )

        if bundle.status not in ("twilio-approved", "provisionally-approved"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Bundle must be approved before purchasing numbers. "
                    f"Current status: {bundle.status}"
                ),
            )

        # Purchase the number with the bundle
        purchased_number = client.incoming_phone_numbers.create(
            phone_number=phone_number,
            friendly_name=friendly_name,
            bundle_sid=bundle_sid,
        )

        # Persist to local DB
        await create_phone_number(
            db,
            org_id=str(org),
            phone_number=purchased_number.phone_number,
            friendly_name=purchased_number.friendly_name or phone_number,
            country_code="",
            twilio_sid=purchased_number.sid,
        )

        return {
            "success": True,
            "phone_number": purchased_number.phone_number,
            "sid": purchased_number.sid,
            "friendly_name": purchased_number.friendly_name,
            "bundle_sid": bundle_sid,
            "bundle_status": bundle.status,
            "compliance_verified": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error purchasing phone number with compliance: %s", e
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to purchase phone number with compliance: "
                f"{e}"
            ),
        )


# ---------------------------------------------------------------------------
# POST /cancel-number
# ---------------------------------------------------------------------------

@router.post("/cancel-number")
async def cancel_phone_number(
    sid: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Cancel / release a phone number from the sub-account."""
    try:
        # -- Verify ownership and get subscription info ------------------
        # Try legacy PhoneNumber table first
        phone_record = None
        stripe_subscription_id = None

        try:
            phone_record = await get_legacy_phone_number_by_sid(db, sid, org)
            if phone_record:
                stripe_subscription_id = phone_record.get(
                    "stripeSubscriptionId"
                )
        except Exception:
            pass

        if not phone_record:
            # Fall back to new phone_numbers table
            phone_record = await get_phone_number_by_twilio_sid(db, sid, org)

        if not phone_record:
            raise HTTPException(
                status_code=403,
                detail="You don't own this phone number",
            )

        # -- Cancel Stripe subscription if it exists ---------------------
        if stripe_subscription_id:
            try:
                stripe.Subscription.delete(stripe_subscription_id)
            except Exception as e:
                logger.error("Error canceling Stripe subscription: %s", e)
                # Continue with number cancellation regardless

        # -- Delete from Twilio ------------------------------------------
        client = await _get_twilio_client(db, org)
        client.incoming_phone_numbers(sid).delete()

        # -- Delete from database(s) ------------------------------------
        try:
            await delete_legacy_phone_number(db, sid)
        except Exception:
            pass  # Legacy table may not exist

        await delete_phone_number_by_sid(db, sid, str(org))

        return {"success": True}

    except TwilioRestException as e:
        logger.error("Twilio API error: %s", e)
        raise HTTPException(status_code=e.status, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Internal server error: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}"
        )


# ---------------------------------------------------------------------------
# POST /update-number
# ---------------------------------------------------------------------------

@router.post("/update-number")
async def update_phone_number(
    payload: UpdateNumberPayload,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Update a phone number's friendly name."""
    try:
        # -- Verify ownership --------------------------------------------
        ownership = None
        try:
            ownership = await check_ownership_legacy(db, payload.sid, org)
        except Exception:
            pass

        if not ownership:
            ownership = await check_ownership_new(db, payload.sid, org)

        if not ownership:
            raise HTTPException(
                status_code=403,
                detail="You don't own this phone number",
            )

        # -- Update in Twilio --------------------------------------------
        client = await _get_twilio_client(db, org)
        updated_number = client.incoming_phone_numbers(payload.sid).update(
            friendly_name=payload.friendly_name,
        )

        # -- Update in legacy DB -----------------------------------------
        try:
            await update_legacy_friendly_name(
                db, payload.friendly_name, payload.sid, org,
            )
        except Exception:
            pass

        # -- Update in new DB table --------------------------------------
        try:
            await update_phone_number_friendly_name(
                db, payload.friendly_name, payload.sid, org,
            )
        except Exception:
            pass

        return {
            "success": True,
            "phone_number": updated_number.phone_number,
            "friendly_name": updated_number.friendly_name,
        }

    except TwilioRestException as e:
        logger.error("Twilio API error: %s", e)
        raise HTTPException(
            status_code=e.status, detail=f"Twilio API error: {e}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Internal server error: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}"
        )
