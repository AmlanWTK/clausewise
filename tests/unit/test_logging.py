"""Unit tests for logging configuration."""

import structlog

from clausewise.config import Settings
from clausewise.observability import configure_logging, get_logger


def test_configure_and_log_does_not_raise() -> None:
    configure_logging(Settings(_env_file=None))
    log = get_logger(__name__)
    log.info("test_event", key="value")


def test_production_uses_json_renderer() -> None:
    configure_logging(Settings(_env_file=None, environment="production"))
    processors = structlog.get_config()["processors"]
    assert isinstance(processors[-1], structlog.processors.JSONRenderer)
