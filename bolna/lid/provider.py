from bolna.helpers.logger_config import configure_logger

from .sarvam import SarvamLID
from .soniox import SonioxLID

logger = configure_logger(__name__)


class LIDProvider:
    _PROVIDERS = {
        "sarvam": SarvamLID,
        "soniox": SonioxLID,
    }

    @classmethod
    def create(cls, provider, on_language, config):
        klass = cls._PROVIDERS.get(provider.lower())
        if klass is None:
            logger.warning(f"LIDProvider: unknown provider '{provider}', falling back to sarvam")
            klass = SarvamLID
        return klass(on_language=on_language, config=config)
