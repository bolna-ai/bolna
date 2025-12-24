import asyncio
import json

import aiohttp
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_to_request_log

logger = configure_logger(__name__)


async def trigger_api(url, method, param, api_token, headers_data, meta_info, run_id, **kwargs):
    try:
        request_body, api_params = None, None
        if param:
            if isinstance(param, dict):
                param = json.dumps(param)
            code = compile(param % kwargs, "<string>", "exec")
            exec(code, globals(), kwargs)
            request_body = param % kwargs
            api_params = json.loads(request_body)

        headers = {'Content-Type': 'application/json'}
        content_type = "json"
        if api_token:
            headers['Authorization'] = api_token

        if headers_data and isinstance(headers_data, dict):
            for k, v in headers_data.items():
                headers[k] = v

        if headers.get('Content-Type').lower().startswith('application/x-www-form-urlencoded'):
            content_type = 'form'
        convert_to_request_log(request_body, meta_info , None, "function_call", direction="request", is_cached=False, run_id=run_id)

        await asyncio.sleep(0.7)

        async with aiohttp.ClientSession() as session:
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

            return response_text
    except Exception as e:
        message = f"ERROR CALLING API: Please check your API: {e}"
        logger.error(message)
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
    return {
        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        for k, v in data.items()
    }
