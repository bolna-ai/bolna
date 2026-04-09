import aiohttp
from .base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class WebhookAgent(BaseAgent):
    def __init__(self, webhook_url, payload=None):
        super().__init__()
        self.webhook_url = webhook_url
        self.payload = payload or {}

    async def __send_payload(self, payload):
        try:
            logger.info(f"Sending a webhook post request {payload}")
            async with aiohttp.ClientSession() as session:
                if payload is not None:
                    async with session.post(self.webhook_url, json=payload) as response:
                        if response.status == 200:
                            # need to check if the returned response is json or not
                            # data = await response.json()
                            return True
                        else:
                            logger.error(f"Error: {response.status} - {await response.text()}")
                            return None
                else:
                    logger.info("Payload was null")
            return None
        except Exception as e:
            logger.error(f"Something went wrong with webhook {self.webhook_url}, {payload}, {str(e)}")

    async def execute(self, payload):
        if not self.webhook_url:
            return None
        response = await self.__send_payload(payload)
        logger.info(f"Response {response}")
        return response
