"""
Logging configuration utility.
Provides a centralized way to set up and retrieve loggers with consistent formatting.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with the given name.
    Ensures that the logger format and level is consistent across modules.
    """
    # Create logger
    logger = logging.getLogger(name)

    # Only configure handlers if the logger doesn't already have them to prevent duplicates
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        logger.addHandler(ch)

        # Optional: prevent propagation to the root logger to avoid duplicate prints
        # if the root logger also has handlers.
        logger.propagate = False

    return logger
