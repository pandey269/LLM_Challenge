"""Centralized configuration for the RAG system."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH, override=False)
else:
    load_dotenv()


class Paths(BaseModel):
    project_root: Path = Field(default=Path(__file__).resolve().parents[2])
    data_dir: Path = Field(default=Path("data"))
    vector_dir: Path = Field(default=Path("data/vectorstore"))
    db_url: str = Field(default="sqlite:///data/metadata.db")


class ModelSettings(BaseModel):
    embed_model: str = Field(default="nomic-ai/nomic-embed-text-v1.5")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_base_url: str = Field(default="http://localhost:11434")
    llm_provider: Literal["ollama", "openai"] = Field(default="openai")
    openai_api_base: str | None = Field(default="https://api.openai.com/v1")
    max_input_tokens: int = Field(default=4096)
    max_output_tokens: int = Field(default=1024)


class RAGSettings(BaseModel):
    chunk_size_tokens: int = Field(default=600)
    chunk_overlap_tokens: int = Field(default=120)
    top_k_dense: int = Field(default=6)
    top_k_sparse: int = Field(default=4)
    reflection_threshold: float = Field(default=0.65)
    max_reflection_loops: int = Field(default=2)


class ObservabilitySettings(BaseModel):
    enable_tracing: bool = Field(default=True)
    otlp_endpoint: str = Field(default="http://localhost:4318")
    enable_prometheus: bool = Field(default=True)


class CacheSettings(BaseModel):
    enabled: bool = Field(default=False)
    path: Path = Field(default=Path("data/cache/lc_cache.db"))


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    environment: str = Field(default="local")
    debug: bool = Field(default=False)
    paths: Paths = Paths()
    model: ModelSettings = ModelSettings()
    rag: RAGSettings = RAGSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    cache: CacheSettings = CacheSettings()


@lru_cache
def get_settings() -> AppSettings:
    """Return cached settings instance."""

    settings = AppSettings()
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.vector_dir.mkdir(parents=True, exist_ok=True)
    cache_path = settings.cache.path
    if not cache_path.is_absolute():
        cache_path = settings.paths.project_root / cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    settings.cache.path = cache_path
    return settings


settings = get_settings()
