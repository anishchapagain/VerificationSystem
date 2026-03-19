"""
Centralized Logging Configuration
===================================
Module  : core/logger.py
Purpose : Configure loguru-based structured logging for the entire application.

Author  : Signature Verifier Team
Version : 1.0.0
"""

import sys
from pathlib import Path
from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

LOG_FORMAT_FILE = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)


def setup_logger(log_level: str = "INFO", enable_json: bool = False, log_to_file: bool = True) -> None:
    """
    Initialize and configure the application logger.

    Sets up three log sinks:
      1. Console (stderr)  — colored, human-readable
      2. Rotating file     — plain text, 10 MB rotation, 7-day retention
      3. Error-only file   — errors and above, 30-day retention
      4. JSON file         — structured logs for log aggregation (optional)

    Args:
        log_level   (str)  : Minimum log level: DEBUG | INFO | WARNING | ERROR | CRITICAL
        enable_json (bool) : Also write structured JSON logs when True
        log_to_file (bool) : Write logs to rotating files when True
    """
    logger.remove()

    logger.add(
        sys.stderr, format=LOG_FORMAT, level=log_level,
        colorize=True, backtrace=True, diagnose=True,
    )

    if log_to_file:
        logger.add(
            LOG_DIR / "app_{time:YYYY-MM-DD}.log",
            format=LOG_FORMAT_FILE, level=log_level,
            rotation="10 MB", retention="7 days", compression="zip",
            backtrace=True, diagnose=False, enqueue=True,
        )
        logger.add(
            LOG_DIR / "errors_{time:YYYY-MM-DD}.log",
            format=LOG_FORMAT_FILE, level="ERROR",
            rotation="5 MB", retention="30 days", compression="zip",
            backtrace=True, diagnose=False, enqueue=True,
        )

    if enable_json:
        logger.add(
            LOG_DIR / "structured_{time:YYYY-MM-DD}.json",
            level=log_level, serialize=True,
            rotation="10 MB", retention="7 days", compression="zip", enqueue=True,
        )

    logger.info(f"Logger initialized | level={log_level} | file={log_to_file} | json={enable_json}")


def get_logger(name: str):
    """Return a named child logger bound to a module context."""
    return logger.bind(module=name)
