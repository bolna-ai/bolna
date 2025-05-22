import logging
import os
from contextvars import ContextVar

# --- Custom log levels ---
CUSTOM_LEVELS = {
    "TRANSCRIBER": 25,
    "LLM": 35,
    "SYNTHESIZER": 45,
}

for name, level in CUSTOM_LEVELS.items():
    logging.addLevelName(level, name)


VALID_LOGGING_LEVELS = list(CUSTOM_LEVELS.keys()) + ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

log_context_data = ContextVar("log_context_data", default={})


original_factory = logging.getLogRecordFactory()

def dynamic_log_record_factory(*args, **kwargs):
    record = original_factory(*args, **kwargs)
    context = log_context_data.get()
    record.context = " ".join(f"{key}={value}" for key, value in context.items()) or ""
    return record

logging.setLogRecordFactory(dynamic_log_record_factory)


class CustomLogger(logging.Logger):
    def trans(self, msg, *args, **kwargs):
        if self.isEnabledFor(CUSTOM_LEVELS["TRANSCRIBER"]):
            self._log(CUSTOM_LEVELS["TRANSCRIBER"], msg, args, **kwargs)

    def llm(self, msg, *args, **kwargs):
        if self.isEnabledFor(CUSTOM_LEVELS["LLM"]):
            self._log(CUSTOM_LEVELS["LLM"], msg, args, **kwargs)

    def synt(self, msg, *args, **kwargs):
        if self.isEnabledFor(CUSTOM_LEVELS["SYNTHESIZER"]):
            self._log(CUSTOM_LEVELS["SYNTHESIZER"], msg, args, **kwargs)


# --- Level filter based on LOGGING_LEVELS env var ---
class LevelWhitelistFilter(logging.Filter):
    def __init__(self, allowed_levels):
        self.allowed_levels = allowed_levels

    def filter(self, record):
        return record.levelno in self.allowed_levels


def parse_allowed_levels():
    raw = os.getenv("LOGGING_LEVELS", "").strip()

 
    if not raw:
        return set(range(0, 100))  

    level_names = [name.strip().upper() for name in raw.split(",") if name.strip()]
    allowed = set()

    for name in level_names:
        if name in CUSTOM_LEVELS:
            allowed.add(CUSTOM_LEVELS[name])
        elif hasattr(logging, name):
            allowed.add(getattr(logging, name))

    return allowed

# --- Configure logger ---
def configure_logger(file_name, logging_level='INFO'):
    logging.setLoggerClass(CustomLogger)

    logger = logging.getLogger(file_name)

    if logger.handlers:
        return logger

    allowed_levels = parse_allowed_levels()
    if not allowed_levels:
        if logging_level not in VALID_LOGGING_LEVELS:
            logging_level = "INFO"
        level = CUSTOM_LEVELS.get(logging_level.upper(), getattr(logging, logging_level.upper(), logging.INFO))
        allowed_levels = {level}

    logger.setLevel(1)

    handler = logging.StreamHandler()
    handler.setLevel(1)
    handler.addFilter(LevelWhitelistFilter(allowed_levels))

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(context)s {%(module)s} [%(funcName)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def set_log_context(key: str, value: str):
    """
    Set a key-value pair in the context.
    """
    ctx = log_context_data.get()
    ctx = ctx.copy()
    ctx[key] = value
    log_context_data.set(ctx)


def get_log_context(key: str) -> str:
    """
    Get a value by key from the context. Returns 'N/A' if key is not found.
    """
    return log_context_data.get().get(key, "")


def clear_log_context():
    """
    Clear the entire context.
    """
    log_context_data.set({})
