import logging
import os
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler


# --- Custom log levels ---
CUSTOM_LEVELS = {
    "TRANSCRIBER": 25,
    "LLM": 35,
    "SYNTHESIZER": 45,
}

COMPONENT_LEVELS = {
    "transcriber": CUSTOM_LEVELS["TRANSCRIBER"],
    "llm": CUSTOM_LEVELS["LLM"],
    "synthesizer": CUSTOM_LEVELS["SYNTHESIZER"],
}

for name, level in CUSTOM_LEVELS.items():
    logging.addLevelName(level, name)

log_context_data = ContextVar("log_context_data", default={})

LOGS_DIR = os.getenv("LOGS_DIR", os.path.join(os.getcwd(), "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)

MAX_LOG_SIZE = int(os.getenv("MAX_LOG_SIZE", 10 * 1024 * 1024))
BACKUP_COUNT = int(os.getenv("BACKUP_LOG_COUNT", 5))

original_factory = logging.getLogRecordFactory()

def dynamic_log_record_factory(*args, **kwargs):
    record = original_factory(*args, **kwargs)
    context = log_context_data.get()
    record.context = " ".join(f"{key}={value}" for key, value in context.items()) or ""

    if record.levelno == CUSTOM_LEVELS["TRANSCRIBER"]:
        record.component = "transcriber"
    elif record.levelno == CUSTOM_LEVELS["LLM"]:
        record.component = "llm"
    elif record.levelno == CUSTOM_LEVELS["SYNTHESIZER"]:
        record.component = "synthesizer"
    else:
        record.component = "general"

    return record

logging.setLogRecordFactory(dynamic_log_record_factory)

class ComponentFilter(logging.Filter):
    def __init__(self, component):
        super().__init__()
        self.component = component

    def filter(self, record):
        return getattr(record, 'component', 'general') == self.component

class LevelWhitelistFilter(logging.Filter):
    def __init__(self, allowed_levels):
        self.allowed_levels = allowed_levels

    def filter(self, record):
        return record.levelno in self.allowed_levels

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

def get_file_handler(log_file, component=None):
    handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(context)s {%(module)s} [%(funcName)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    if component:
        handler.addFilter(ComponentFilter(component))

    return handler

def configure_logger(file_name):
    logging.setLoggerClass(CustomLogger)
    logger = logging.getLogger("file_name")

    if logger.handlers:
        return logger

    logger.setLevel(1)

    # Default to include only these levels
    allowed_levels = {
        logging.INFO,
        logging.ERROR,
        CUSTOM_LEVELS["TRANSCRIBER"],
        CUSTOM_LEVELS["LLM"],
        CUSTOM_LEVELS["SYNTHESIZER"],
    }

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(context)s {%(module)s} [%(funcName)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(1)
    console_handler.addFilter(LevelWhitelistFilter(allowed_levels))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Main log file - contains all logs
    main_log_file = os.path.join(LOGS_DIR, "all_logs.log")
    main_handler = get_file_handler(main_log_file)
    main_handler.addFilter(LevelWhitelistFilter(allowed_levels))
    logger.addHandler(main_handler)

    # Info log - consolidated for all modules
    info_log = os.path.join(LOGS_DIR, "info_logs.log")
    info_handler = get_file_handler(info_log)
    info_handler.addFilter(LevelWhitelistFilter({logging.INFO}))
    logger.addHandler(info_handler)

    # Error log - consolidated for all modules
    error_log = os.path.join(LOGS_DIR, "error_logs.log")
    error_handler = get_file_handler(error_log)
    error_handler.addFilter(LevelWhitelistFilter({logging.ERROR}))
    logger.addHandler(error_handler)

    # Component logs - consolidated for all modules
    for component, level in COMPONENT_LEVELS.items():
        component_log = os.path.join(LOGS_DIR, f"{component}_logs.log")
        component_handler = get_file_handler(component_log, component)
        component_handler.addFilter(LevelWhitelistFilter({level}))
        logger.addHandler(component_handler)

    logger.propagate = False
    return logger

def set_log_context(key: str, value: str):
    ctx = log_context_data.get()
    ctx = ctx.copy()
    ctx[key] = value
    log_context_data.set(ctx)

def get_log_context(key: str) -> str:
    return log_context_data.get().get(key, "")

def clear_log_context():
    log_context_data.set({})
