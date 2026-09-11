from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
TARGET_CITY = "澳门"
TARGET_COUNTRY = "中国"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        env_prefix="TRAVEL_AGENT_",
        extra="ignore",
    )

    app_name: str = "Macao Travel Agent API"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"

    database_url: str = (
        "postgresql+asyncpg://travel_agent:travel_agent_dev@127.0.0.1:5432/travel_agent"
    )
    jwt_secret_key: str
    access_token_expire_minutes: int = 30

    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    embedding_model: str = "qwen3-embedding:4b"
    tavily_api_key: str = ""

    chroma_host: str = "127.0.0.1"
    chroma_port: int = 8001
    chroma_collection: str = "macau_travel_knowledge"

    rag_top_k: int = 5
    rag_score_threshold: float = 0.2


@lru_cache
def get_settings() -> Settings:
    return Settings()
