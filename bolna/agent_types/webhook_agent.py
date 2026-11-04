import aiohttp
from .base_agent import BaseAgent
from bolna.helpers.function_calling_helpers import SSRFError, validate_outbound_url
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

WEBHOOK_TIMEOUT_SECONDS = 10


class WebhookAgent(BaseAgent):
    def __init__(self, webhook_url, payload=None):
        super().__init__()
        self.webhook_url = webhook_url
        self.payload = payload or {}

    async def __send_payload(self, payload):
        try:
            logger.info(f"Sending a webhook post request {payload}")
            if payload is None:
                logger.info("Payload was null")
                return None

            await validate_outbound_url(self.webhook_url)
            timeout = aiohttp.ClientTimeout(total=WEBHOOK_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.webhook_url, json=payload, allow_redirects=False) as response:
                    if response.status == 200:
                        # need to check if the returned response is json or not
                        # data = await response.json()
                        return True
                    logger.error(f"Error: {response.status} - {await response.text()}")
            return None
        except SSRFError as e:
            logger.warning(f"Blocked outbound webhook URL: {e}")
            return None
        except Exception as e:
            logger.error(f"Something went wrong with webhook {self.webhook_url}, {payload}, {str(e)}")
            return None

    async def execute(self, payload):
        if not self.webhook_url:
            return None
        response = await self.__send_payload(payload)
        logger.info(f"Response {response}")
        return response
