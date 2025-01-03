__version__ = "0.9.3"

import os 
from bolna.helpers.logger_config import configure_logger
logger = configure_logger(__name__)


def setenv(variables):
    """
    Set environment variables
    """
    for key, value in variables.items():
        logger.info(f"Setting environment variable: {key}")
        os.environ[key] = value