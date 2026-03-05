"""Unit tests for the centralized logging utility."""

import logging
from src.logging_utils import get_logger


def test_get_logger_creation():
    """Test that get_logger creates a logger with correct name and level."""
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert not logger.propagate


def test_get_logger_no_duplicate_handlers():
    """Test that calling get_logger multiple times does not duplicate handlers."""
    logger1 = get_logger("test_duplicate")
    assert len(logger1.handlers) == 1

    logger2 = get_logger("test_duplicate")
    assert logger1 is logger2
    assert len(logger2.handlers) == 1  # Should still be 1, not 2
