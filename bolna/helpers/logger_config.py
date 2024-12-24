import logging
from contextvars import ContextVar


VALID_LOGGING_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

log_context_data = ContextVar("log_context_data", default={})

original_factory = logging.getLogRecordFactory()


def dynamic_log_record_factory(*args, **kwargs):
    record = original_factory(*args, **kwargs)
    # Inject the entire context dictionary as a single attribute
    context = log_context_data.get()
    record.context = " ".join(f"{{{key}={value}}}" for key, value in context.items()) or ""
    return record


logging.setLogRecordFactory(dynamic_log_record_factory)


def configure_logger(file_name, logging_level='INFO'):
    if logging_level not in VALID_LOGGING_LEVELS:
        logging_level = "INFO"

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(context)s {%(module)s} [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(file_name)
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
