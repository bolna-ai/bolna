import asyncio
import json

import aiohttp
from bolna.helpers.logger_config import configure_logger
from bolna.enums import LogComponent, LogDirection
from bolna.helpers.utils import convert_to_request_log, format_error_message

logger = configure_logger(__name__)


def _contains_var_markers(obj):
    """
    Check if object contains any {"$var": ...} markers.

    Args:
        obj: JSON object (dict, list, or primitive)

    Returns:
        True if $var markers are found, False otherwise
    """
    if isinstance(obj, dict):
        if "$var" in obj:
            return True
        return any(_contains_var_markers(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(_contains_var_markers(item) for item in obj)
    return False


def substitute_var_markers(obj, values):
    """
    Recursively substitute {"$var": "name"} markers with actual values.

    This provides type-safe JSON substitution where arrays remain arrays
    and objects remain objects, without string manipulation.

    Args:
        obj: JSON object (dict, list, or primitive)
        values: dict of variable names to values

    Returns:
        Object with markers replaced by actual values

    Example:
        obj = {"products": {"$var": "products"}, "static": "value"}
        values = {"products": [{"code": "123"}]}
        result = {"products": [{"code": "123"}], "static": "value"}
    """
    if isinstance(obj, dict):
        # Check if this is a $var marker
        if len(obj) == 1 and "$var" in obj:
            var_name = obj["$var"]
            if var_name in values:
                return values[var_name]
            else:
                # Keep marker if no value provided (for debugging)
                logger.warning(f"No value provided for $var marker: {var_name}")
                return obj
        # Recursively process dict values
        return {k: substitute_var_markers(v, values) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_var_markers(item, values) for item in obj]
    else:
        return obj  # Primitives returned as-is


def prepare_api_request(param, api_token, headers_data, **kwargs):
    request_body, api_params = None, None
    if param:
        # NEW FORMAT: Check for $var markers (type-safe JSON substitution)
        if isinstance(param, dict) and _contains_var_markers(param):
            api_params = substitute_var_markers(param, kwargs)
            request_body = json.dumps(api_params)
            logger.info("Using $var marker substitution for param")
        else:
            # LEGACY FORMAT: String template with %(field)s placeholders
            if isinstance(param, dict):
                param = json.dumps(param)

            # JSON-serialize complex values (lists, dicts) for proper string substitution
            # Python's % formatting uses repr() which produces single quotes,
            # but JSON requires double quotes for valid JSON output
            json_kwargs = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in kwargs.items()}

            code = compile(param % json_kwargs, "<string>", "exec")
            exec(code, globals(), json_kwargs)
            request_body = param % json_kwargs
            api_params = json.loads(request_body)

    headers = {"Content-Type": "application/json"}
    content_type = "json"
    if api_token:
        headers["Authorization"] = api_token

    if headers_data and isinstance(headers_data, dict):
        for k, v in headers_data.items():
            headers[k] = v

    if headers.get("Content-Type").lower().startswith("application/x-www-form-urlencoded"):
        content_type = "form"

    return {
        "request_body": request_body,
        "api_params": api_params,
        "headers": headers,
        "content_type": content_type,
    }


async def trigger_api(
    url, method, param, api_token, headers_data, meta_info, run_id, return_response_metadata=False, **kwargs
):
    try:
        prepared_request = prepare_api_request(param, api_token, headers_data, **kwargs)
        request_body = prepared_request["request_body"]
        api_params = prepared_request["api_params"]
        headers = prepared_request["headers"]
        content_type = prepared_request["content_type"]
        convert_to_request_log(
            request_body,
            meta_info,
            None,
            LogComponent.FUNCTION_CALL,
            direction=LogDirection.REQUEST,
            is_cached=False,
            run_id=run_id,
        )
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            if method.lower() == "get":
                logger.info(f"Sending request {request_body}, {url}, {headers}")
                async with session.get(url, params=api_params, headers=headers) as response:
                    response_text = await response.text()
            elif method.lower() == "post":
                logger.info(f"Sending request {api_params}, {url}, {headers}")
                if content_type == "json":
                    async with session.post(url, json=api_params, headers=headers) as response:
                        response_text = await response.text()
                elif content_type == "form":
                    normalized_api_params = normalize_for_form(api_params)
                    async with session.post(url, data=normalized_api_params, headers=headers) as response:
                        response_text = await response.text()

            if return_response_metadata:
                return {
                    "status_code": response.status if response is not None else None,
                    "body": response_text,
                    "content_type": response.headers.get("Content-Type") if response is not None else None,
                }

            return response_text
    except asyncio.TimeoutError:
        message = f"ERROR CALLING API: Request to {url} timed out after 5 seconds"
        logger.error(message)
        if run_id:
            convert_to_request_log(
                format_error_message("function_call", url, "Timed out after 5 seconds"),
                meta_info,
                model=None,
                component=LogComponent.WARNING,
                direction=LogDirection.WARNING,
                is_cached=False,
                run_id=run_id,
            )
        if return_response_metadata:
            return {
                "status_code": None,
                "body": message,
                "content_type": None,
                "error": "Timed out after 5 seconds",
            }
        return message
    except Exception as e:
        message = f"ERROR CALLING API: Please check your API: {e}"
        logger.error(message)
        if run_id:
            convert_to_request_log(
                format_error_message("function_call", url, str(e)),
                meta_info,
                model=None,
                component=LogComponent.WARNING,
                direction=LogDirection.WARNING,
                is_cached=False,
                run_id=run_id,
            )
        if return_response_metadata:
            return {
                "status_code": None,
                "body": message,
                "content_type": None,
                "error": str(e),
            }
        return message


async def computed_api_response(response):
    get_res_keys, get_res_values = None, None
    try:
        get_res_keys = list(json.loads(response).keys())
        get_res_values = list(json.loads(response).values())
    except Exception as e:
        pass

    return get_res_keys, get_res_values


def normalize_for_form(data: dict) -> dict:
    return {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in data.items()}
