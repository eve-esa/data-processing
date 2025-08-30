# refer - https://loguru.readthedocs.io/en/stable/api/logger.html
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

logger.remove() # remove default stuff

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)


# store error log files
logger.add(
    log_dir / "errors_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    rotation="5 MB",
    retention="90 days",
)

# Function to get logger with context
def get_logger(name: Optional[str] = None):
    if name:
        return logger.bind(name=name)
    return logger


def add_log_file(filepath: str, level: str = "INFO", **kwargs):
    logger.add(filepath, level=level, **kwargs)


def set_log_level(level: str):
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)
