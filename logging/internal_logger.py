import logging
import sys
from pathlib import Path
from typing import Optional


def setup_internal_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    filename: str = "internal.log",
) -> logging.Logger:
    """
    Create a reusable internal project logger.

    - Console logs: INFO+
    - File logs: DEBUG+
    - Avoids duplicate handlers
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            Path(log_dir) / filename,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger