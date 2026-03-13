"""
Country / phone-number availability and compliance routes.

main_server.py reference: lines ~8762-9256.

Routes (exact paths from main_server.py):
  GET  /countries/available                    -- list all countries where Twilio numbers are available
  GET  /countries/{country_code}/number-types  -- available number types for a country
  GET  /countries/{country_code}/compliance    -- compliance requirements for a country
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.dependencies import DBDep, OrgDep
from db.queries.organizations import get_org_by_id

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/countries", tags=["countries"])


# ---------------------------------------------------------------------------
# Helpers -- Twilio sub-account client
# ---------------------------------------------------------------------------

async def _get_sub_client(db, org_id):
    """Return a Twilio Client configured with the org's sub-account creds."""
    from twilio.rest import Client  # lazy import

    row = await get_org_by_id(db, str(org_id))
    if not row or not row["twilio_subaccount_sid"] or not row["twilio_subaccount_auth_token"]:
        raise HTTPException(
            status_code=400,
            detail="Twilio credentials not configured for this account",
        )
    return Client(row["twilio_subaccount_sid"], row["twilio_subaccount_auth_token"])


# ---------------------------------------------------------------------------
# Helpers -- compliance level classification
# ---------------------------------------------------------------------------

_HIGH_COMPLIANCE = frozenset([
    "IN", "CN", "BR", "JP", "KR", "SA", "AE", "EG", "VE", "MY", "ID", "TH", "VN",
])

_MEDIUM_COMPLIANCE = frozenset([
    "GB", "DE", "FR", "IT", "ES", "NL", "BE", "AT", "CH", "SE", "NO", "DK", "FI", "PL", "CZ", "HU",
])


def _determine_compliance_level(country_code: str) -> str:
    """Classify a country into low / medium / high compliance."""
    if country_code in _HIGH_COMPLIANCE:
        return "high"
    if country_code in _MEDIUM_COMPLIANCE:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Helpers -- request-only countries (not in Twilio instant API)
# ---------------------------------------------------------------------------

_REQUEST_ONLY_COUNTRIES = [
    {"country_code": "AE", "country": "United Arab Emirates", "available_number_types": ["local", "mobile"], "compliance_level": "high"},
    {"country_code": "SA", "country": "Saudi Arabia", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "QA", "country": "Qatar", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "KW", "country": "Kuwait", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "BH", "country": "Bahrain", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "OM", "country": "Oman", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "JO", "country": "Jordan", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "LB", "country": "Lebanon", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "EG", "country": "Egypt", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "MA", "country": "Morocco", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "TN", "country": "Tunisia", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "DZ", "country": "Algeria", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "KE", "country": "Kenya", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "NG", "country": "Nigeria", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "ZA", "country": "South Africa", "available_number_types": ["local", "mobile"], "compliance_level": "medium"},
    {"country_code": "GH", "country": "Ghana", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "UG", "country": "Uganda", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "TZ", "country": "Tanzania", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "RW", "country": "Rwanda", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "SN", "country": "Senegal", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "CI", "country": "C\u00f4te d'Ivoire", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "CM", "country": "Cameroon", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "MG", "country": "Madagascar", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "ZM", "country": "Zambia", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "ZW", "country": "Zimbabwe", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "BW", "country": "Botswana", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "MU", "country": "Mauritius", "available_number_types": ["mobile"], "compliance_level": "low"},
    {"country_code": "PK", "country": "Pakistan", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "BD", "country": "Bangladesh", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "LK", "country": "Sri Lanka", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "NP", "country": "Nepal", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "KH", "country": "Cambodia", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "LA", "country": "Laos", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "MM", "country": "Myanmar", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "MN", "country": "Mongolia", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "KZ", "country": "Kazakhstan", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "UZ", "country": "Uzbekistan", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "KG", "country": "Kyrgyzstan", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "TJ", "country": "Tajikistan", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "TM", "country": "Turkmenistan", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "AF", "country": "Afghanistan", "available_number_types": ["mobile"], "compliance_level": "high"},
    {"country_code": "AL", "country": "Albania", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "BA", "country": "Bosnia and Herzegovina", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "BG", "country": "Bulgaria", "available_number_types": ["local", "mobile"], "compliance_level": "medium"},
    {"country_code": "HR", "country": "Croatia", "available_number_types": ["local", "mobile"], "compliance_level": "medium"},
    {"country_code": "ME", "country": "Montenegro", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "MK", "country": "North Macedonia", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "RS", "country": "Serbia", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "SI", "country": "Slovenia", "available_number_types": ["local", "mobile"], "compliance_level": "medium"},
    {"country_code": "SK", "country": "Slovakia", "available_number_types": ["local", "mobile"], "compliance_level": "medium"},
    {"country_code": "LV", "country": "Latvia", "available_number_types": ["local", "mobile"], "compliance_level": "medium"},
    {"country_code": "LT", "country": "Lithuania", "available_number_types": ["local", "mobile"], "compliance_level": "medium"},
    {"country_code": "EE", "country": "Estonia", "available_number_types": ["local", "mobile"], "compliance_level": "medium"},
    {"country_code": "CY", "country": "Cyprus", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "MT", "country": "Malta", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "IS", "country": "Iceland", "available_number_types": ["local"], "compliance_level": "low"},
    {"country_code": "LU", "country": "Luxembourg", "available_number_types": ["local"], "compliance_level": "medium"},
    {"country_code": "LI", "country": "Liechtenstein", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "MC", "country": "Monaco", "available_number_types": ["mobile"], "compliance_level": "medium"},
    {"country_code": "AD", "country": "Andorra", "available_number_types": ["mobile"], "compliance_level": "medium"},
]


# ---------------------------------------------------------------------------
# Helpers -- fallback / demo data
# ---------------------------------------------------------------------------

def _get_enhanced_demo_countries() -> dict:
    """Return enhanced demo countries data when the Twilio API is unavailable."""
    return {
        "countries": [
            {
                "country_code": "US",
                "country": "United States",
                "beta": False,
                "available_number_types": ["local", "national", "mobile", "toll-free"],
                "compliance_level": "low",
                "availability_type": "instant",
            },
            {
                "country_code": "CA",
                "country": "Canada",
                "beta": False,
                "available_number_types": ["local", "toll-free"],
                "compliance_level": "low",
                "availability_type": "instant",
            },
            {
                "country_code": "GB",
                "country": "United Kingdom",
                "beta": False,
                "available_number_types": ["local", "national", "mobile"],
                "compliance_level": "medium",
                "availability_type": "instant",
            },
            {
                "country_code": "DE",
                "country": "Germany",
                "beta": False,
                "available_number_types": ["local", "national"],
                "compliance_level": "medium",
                "availability_type": "instant",
            },
            {
                "country_code": "AU",
                "country": "Australia",
                "beta": False,
                "available_number_types": ["local", "mobile", "toll-free"],
                "compliance_level": "medium",
                "availability_type": "instant",
            },
            {
                "country_code": "IN",
                "country": "India",
                "beta": False,
                "available_number_types": ["mobile"],
                "compliance_level": "high",
                "availability_type": "instant",
            },
            {
                "country_code": "AE",
                "country": "United Arab Emirates",
                "beta": False,
                "available_number_types": ["local", "mobile"],
                "compliance_level": "high",
                "availability_type": "request_only",
            },
            {
                "country_code": "SA",
                "country": "Saudi Arabia",
                "beta": False,
                "available_number_types": ["mobile"],
                "compliance_level": "high",
                "availability_type": "request_only",
            },
            {
                "country_code": "QA",
                "country": "Qatar",
                "beta": False,
                "available_number_types": ["mobile"],
                "compliance_level": "high",
                "availability_type": "request_only",
            },
            {
                "country_code": "KW",
                "country": "Kuwait",
                "beta": False,
                "available_number_types": ["mobile"],
                "compliance_level": "high",
                "availability_type": "request_only",
            },
        ],
        "total_count": 10,
        "instant_available": 6,
        "request_only": 4,
    }


def _get_demo_number_types(country_code: str) -> dict:
    """Return demo number-type data when the Twilio API is unavailable."""
    demo_types = {
        "US": {
            "local":     {"available": True,  "name": "Local Numbers",     "description": "Numbers assigned to a specific geographic region"},
            "mobile":    {"available": True,  "name": "Mobile Numbers",    "description": "Mobile/cellular numbers"},
            "national":  {"available": True,  "name": "National Numbers",  "description": "Non-geographic numbers reachable nationwide"},
            "toll-free": {"available": True,  "name": "Toll-Free Numbers", "description": "Free for callers, you pay for incoming calls"},
        },
        "CA": {
            "local":     {"available": True,  "name": "Local Numbers",     "description": "Numbers assigned to a specific geographic region"},
            "mobile":    {"available": False, "name": "Mobile Numbers",    "description": "Mobile/cellular numbers"},
            "national":  {"available": False, "name": "National Numbers",  "description": "Non-geographic numbers reachable nationwide"},
            "toll-free": {"available": True,  "name": "Toll-Free Numbers", "description": "Free for callers, you pay for incoming calls"},
        },
        "GB": {
            "local":     {"available": True,  "name": "Local Numbers",     "description": "Numbers assigned to a specific geographic region"},
            "mobile":    {"available": True,  "name": "Mobile Numbers",    "description": "Mobile/cellular numbers"},
            "national":  {"available": True,  "name": "National Numbers",  "description": "Non-geographic numbers reachable nationwide"},
            "toll-free": {"available": False, "name": "Toll-Free Numbers", "description": "Free for callers, you pay for incoming calls"},
        },
    }

    default_types = {
        "local":     {"available": True,  "name": "Local Numbers",     "description": "Numbers assigned to a specific geographic region"},
        "mobile":    {"available": False, "name": "Mobile Numbers",    "description": "Mobile/cellular numbers"},
        "national":  {"available": False, "name": "National Numbers",  "description": "Non-geographic numbers reachable nationwide"},
        "toll-free": {"available": False, "name": "Toll-Free Numbers", "description": "Free for callers, you pay for incoming calls"},
    }

    return {
        "country_code": country_code,
        "available_types": demo_types.get(country_code, default_types),
    }


def _get_fallback_compliance(country_code: str) -> dict:
    """Return fallback compliance info when the Twilio regulatory API is unavailable."""
    compliance_level = _determine_compliance_level(country_code)
    return {
        "country_code": country_code,
        "has_regulations": compliance_level != "low",
        "regulations": [],
        "compliance_level": compliance_level,
    }


# ===========================================================================
# GET /countries/available
# ===========================================================================

@router.get("/available")
async def get_available_countries(org: OrgDep, db: DBDep) -> dict:
    """Get all countries where Twilio phone numbers are available.

    Returns both *instant* countries (queryable via Twilio's
    ``AvailablePhoneNumbers`` API) and *request-only* countries that require
    special approval.
    """
    try:
        from twilio.base.exceptions import TwilioRestException

        client = await _get_sub_client(db, org)
        countries = client.available_phone_numbers.list()

        formatted_countries: list[dict] = []

        # Process Twilio API countries (instant availability)
        for country in countries:
            try:
                country_info: dict = {
                    "country_code": country.country_code,
                    "country": country.country,
                    "uri": country.uri,
                    "beta": getattr(country, "beta", False),
                    "subresource_uris": country.subresource_uris,
                    "availability_type": "instant",
                }

                # Determine available number types from sub-resource URIs
                available_types: list[str] = []
                if "local" in country.subresource_uris:
                    available_types.append("local")
                if "mobile" in country.subresource_uris:
                    available_types.append("mobile")
                if "national" in country.subresource_uris:
                    available_types.append("national")
                if "toll_free" in country.subresource_uris:
                    available_types.append("toll-free")

                country_info["available_number_types"] = available_types
                country_info["compliance_level"] = _determine_compliance_level(country.country_code)

                formatted_countries.append(country_info)
            except Exception as exc:
                logger.warning("Error processing country %s: %s", country.country_code, exc)
                continue

        # Merge in request-only countries that were not already returned
        existing_codes = {c["country_code"] for c in formatted_countries}
        for req_country in _REQUEST_ONLY_COUNTRIES:
            if req_country["country_code"] not in existing_codes:
                entry = dict(req_country)  # shallow copy
                entry.update({
                    "beta": False,
                    "subresource_uris": {},
                    "availability_type": "request_only",
                    "uri": (
                        f"/2010-04-01/Accounts/"
                        f"{settings.twilio_account_sid}/AvailablePhoneNumbers/"
                        f"{req_country['country_code']}.json"
                    ),
                })
                formatted_countries.append(entry)

        # Sort alphabetically by country name
        formatted_countries.sort(key=lambda x: x["country"])

        return {
            "countries": formatted_countries,
            "total_count": len(formatted_countries),
            "instant_available": sum(
                1 for c in formatted_countries if c.get("availability_type") == "instant"
            ),
            "request_only": sum(
                1 for c in formatted_countries if c.get("availability_type") == "request_only"
            ),
        }

    except TwilioRestException as exc:
        logger.warning("Twilio API error, returning enhanced demo countries: %s", exc)
        return _get_enhanced_demo_countries()
    except Exception as exc:
        logger.warning("Error fetching countries, returning enhanced demo data: %s", exc)
        return _get_enhanced_demo_countries()


# ===========================================================================
# GET /countries/{country_code}/number-types
# ===========================================================================

@router.get("/{country_code}/number-types")
async def get_country_number_types(
    country_code: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Get available number types (local, mobile, national, toll-free)
    for a specific country by probing the Twilio API for each type.
    """
    try:
        from twilio.base.exceptions import TwilioRestException

        client = await _get_sub_client(db, org)
        country_numbers = client.available_phone_numbers(country_code)

        available_types: dict = {}

        # -- local ---------------------------------------------------------
        try:
            local_sample = country_numbers.local.list(limit=1)
            available_types["local"] = {
                "available": len(local_sample) > 0,
                "name": "Local Numbers",
                "description": "Numbers assigned to a specific geographic region",
            }
        except Exception:
            available_types["local"] = {
                "available": False,
                "name": "Local Numbers",
                "description": "Numbers assigned to a specific geographic region",
            }

        # -- mobile --------------------------------------------------------
        try:
            mobile_sample = country_numbers.mobile.list(limit=1)
            available_types["mobile"] = {
                "available": len(mobile_sample) > 0,
                "name": "Mobile Numbers",
                "description": "Mobile/cellular numbers",
            }
        except Exception:
            available_types["mobile"] = {
                "available": False,
                "name": "Mobile Numbers",
                "description": "Mobile/cellular numbers",
            }

        # -- national ------------------------------------------------------
        try:
            national_sample = country_numbers.national.list(limit=1)
            available_types["national"] = {
                "available": len(national_sample) > 0,
                "name": "National Numbers",
                "description": "Non-geographic numbers reachable nationwide",
            }
        except Exception:
            available_types["national"] = {
                "available": False,
                "name": "National Numbers",
                "description": "Non-geographic numbers reachable nationwide",
            }

        # -- toll-free -----------------------------------------------------
        try:
            tollfree_sample = country_numbers.toll_free.list(limit=1)
            available_types["toll-free"] = {
                "available": len(tollfree_sample) > 0,
                "name": "Toll-Free Numbers",
                "description": "Free for callers, you pay for incoming calls",
            }
        except Exception:
            available_types["toll-free"] = {
                "available": False,
                "name": "Toll-Free Numbers",
                "description": "Free for callers, you pay for incoming calls",
            }

        return {
            "country_code": country_code,
            "available_types": available_types,
        }

    except TwilioRestException as exc:
        logger.warning("Twilio API error for %s: %s", country_code, exc)
        return _get_demo_number_types(country_code)
    except Exception as exc:
        logger.warning("Error checking number types for %s: %s", country_code, exc)
        return _get_demo_number_types(country_code)


# ===========================================================================
# GET /countries/{country_code}/compliance
# ===========================================================================

@router.get("/{country_code}/compliance")
async def get_country_compliance_info(
    country_code: str,
    org: OrgDep,
    db: DBDep,
) -> dict:
    """Get compliance / regulatory requirements for a specific country.

    Queries Twilio's ``regulatory_compliance.regulations`` API and assigns
    a complexity-based compliance level (low / medium / high).
    """
    try:
        from twilio.base.exceptions import TwilioRestException

        try:
            client = await _get_sub_client(db, org)
            regulations = (
                client.numbers.v2.regulatory_compliance
                .regulations.list(iso_country=country_code)
            )

            compliance_info: dict = {
                "country_code": country_code,
                "has_regulations": len(regulations) > 0,
                "regulations": [],
                "compliance_level": "low",
            }

            if regulations:
                max_complexity = 0
                for regulation in regulations:
                    reg_info = {
                        "regulation_sid": regulation.sid,
                        "friendly_name": regulation.friendly_name,
                        "end_user_type": getattr(regulation, "end_user_type", None),
                        "number_type": getattr(regulation, "number_type", None),
                    }
                    compliance_info["regulations"].append(reg_info)

                    # Simple complexity scoring
                    complexity = 1
                    name_lower = str(regulation.friendly_name).lower()
                    if "business" in name_lower:
                        complexity += 1
                    if "identity" in name_lower:
                        complexity += 1
                    if "address" in name_lower:
                        complexity += 1

                    max_complexity = max(max_complexity, complexity)

                # Map complexity score to level
                if max_complexity >= 3:
                    compliance_info["compliance_level"] = "high"
                elif max_complexity >= 2:
                    compliance_info["compliance_level"] = "medium"
                else:
                    compliance_info["compliance_level"] = "low"

            return compliance_info

        except TwilioRestException:
            # Twilio regulatory API unavailable -- fall back to static data
            return _get_fallback_compliance(country_code)

    except Exception as exc:
        logger.warning("Error getting compliance for %s: %s", country_code, exc)
        return _get_fallback_compliance(country_code)
