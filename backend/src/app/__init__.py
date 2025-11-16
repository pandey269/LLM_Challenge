"""App package for Intelligent Document Q&A system."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import settings

try:
    from langchain.cache import SQLiteCache
    from langchain.globals import set_llm_cache
except ImportError:  # pragma: no cover - optional dependency
    SQLiteCache = None  # type: ignore[assignment]
    set_llm_cache = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _configure_cache() -> None:
    if not settings.cache.enabled:
        return
    if SQLiteCache is None or set_llm_cache is None:
        logger.warning("LangChain cache not available; install langchain>=0.1 for caching support.")
        return

    cache_path: Path = settings.cache.path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(str(cache_path)))
    logger.info("LangChain cache enabled at %s", cache_path)


_configure_cache()
