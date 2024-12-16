import asyncio
import json

import aiohttp
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_to_request_log

logger = configure_logger(__name__)

async def trigger_api(url, method, param, api_token, meta_info, run_id, header=None, **kwargs):
    try:
        request_body, api_params = None, None

        # Replace placeholders in the URL dynamically
        if "%(" in url and ")s" in url:
            try:
                url = url % kwargs
                logger.info(f"Processed URL with dynamic parameters: {url}")
            except KeyError as e:
                message = f"ERROR: Missing URL parameter: {e}"
                logger.error(message)
                return message

        # Handle request parameters dynamically
        if param:
            try:
                code = compile(param % kwargs, "<string>", "exec")
                exec(code, globals(), kwargs)
                request_body = param % kwargs
                api_params = json.loads(request_body)

                logger.info(f"Params {param % kwargs} \n {type(request_body)} \n {param} \n {kwargs} \n\n {request_body}")
            except Exception as e:
                logger.error(f"Error processing request parameters: {e}")
                return f"ERROR: Invalid parameters: {e}"
        else:
            logger.info(f"Params {param} \n {type(request_body)} \n {param} \n {kwargs} \n\n {request_body}")

        # Default headers setup
        if(header):
            headers = json.loads(header)
        else:
            headers = {'Content-Type': 'application/json'}
        if api_token:
            headers['Authorization'] = api_token

        convert_to_request_log(request_body, meta_info, None, "function_call", direction="request", is_cached=False, run_id=run_id)

        logger.info("Sleeping for 700 ms to make sure that we do not send the same message multiple times")
        await asyncio.sleep(0.7)

        async with aiohttp.ClientSession() as session:
            # Handle different HTTP methods
            if method.lower() == "get":
                logger.info(f"Sending GET request: {url}, Params: {api_params}, Headers: {headers}")
                async with session.get(url, params=api_params, headers=headers) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server: {response_text}")

            elif method.lower() == "post":
                logger.info(f"Sending POST request: {url}, Data: {api_params}, Headers: {headers}")
                async with session.post(url, json=api_params, headers=headers) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server: {response_text}")

            elif method.lower() == "put":
                logger.info(f"Sending PUT request: {url}, Data: {api_params}, Headers: {headers}")
                async with session.put(url, json=api_params, headers=headers) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server: {response_text}")

            elif method.lower() == "delete":
                logger.info(f"Sending DELETE request: {url}, Data: {api_params}, Headers: {headers}")
                async with session.delete(url, json=api_params, headers=headers) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server: {response_text}")

            else:
                logger.error(f"Unsupported HTTP method: {method}")
                response_text = f"Unsupported HTTP method: {method}"

            return response_text

    except Exception as e:
        message = f"ERROR CALLING API: There was an error calling the API: {e}"
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
