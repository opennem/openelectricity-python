"""
Logging configuration for OpenElectricity.

This module configures logging based on the environment settings.
Development environments will show debug logs, while production
environments will only show info and above.
"""

import logging
import sys

from openelectricity.settings_schema import settings


def configure_logging() -> None:
    """Configure root logger based on environment settings."""
    root_logger = logging.getLogger("openelectricity")

    # Clear any existing handlers
    root_logger.handlers = []

    # Set level based on environment
    level = logging.DEBUG if settings.is_development else logging.INFO
    root_logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the openelectricity prefix.

    Args:
        name: The name of the logger, will be prefixed with 'openelectricity.'

    Returns:
        A configured logger instance
    """
    return logging.getLogger(f"openelectricity.{name}")


# Configure logging on module import
configure_logging()
