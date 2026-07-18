"""Structured logging configuration.

- Development: human-readable, colored console output.
- Production: one JSON object per line (machine-parseable, grep/jq-friendly).
- ``structlog.contextvars`` is merged into every event, so request-scoped
  context (e.g. ``request_id``, bound in API middleware later) appears on every
  log line without threading it through call sites.

Call ``configure_logging()`` exactly once per process, at the entry point.
"""

import logging
import sys
from typing import cast

import structlog

from clausewise.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog and the stdlib root logger for this process."""
    level = getattr(logging, settings.log_level)

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: structlog.typing.Processor
    if settings.is_production:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (SQLAlchemy, Alembic, uvicorn, ...) to stderr at the
    # same level so third-party logs are not silently lost.
    logging.basicConfig(stream=sys.stderr, level=level, format="%(levelname)s %(name)s %(message)s")


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Return a named structlog logger. Prefer module-level ``get_logger(__name__)``."""
    # structlog.get_logger is typed as Any by design (proxy object); the configured
    # wrapper_class guarantees a FilteringBoundLogger at runtime.
    return cast("structlog.typing.FilteringBoundLogger", structlog.get_logger(name))
