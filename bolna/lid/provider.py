from bolna.helpers.logger_config import configure_logger

from .azure import AzureLID
from .sarvam import SarvamLID

logger = configure_logger(__name__)


class LIDProvider:
    _PROVIDERS = {
        "sarvam": SarvamLID,
        "azure": AzureLID,
    }

    @classmethod
    def create(cls, provider, on_language, config):
        klass = cls._PROVIDERS.get(provider.lower())
        if klass is None:
            logger.warning(f"LIDProvider: unknown provider '{provider}', falling back to azure")
            klass = AzureLID
        return klass(on_language=on_language, config=config)
