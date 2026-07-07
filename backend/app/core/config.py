from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration, loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "mock"
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "ikios_chunks"
    qdrant_local_path: str = "./.qdrant_local"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ikios_dev_password"

    metadata_db_url: str = "sqlite+aiosqlite:///./ikios_metadata.db"
    learning_db_url: str = "sqlite+aiosqlite:///./ikios_learning.db"
    experiment_db_url: str = "sqlite+aiosqlite:///./ikios_experiments.db"
    ml_registry_db_url: str = "sqlite+aiosqlite:///./ikios_ml_registry.db"
    ml_models_dir: str = "./.ml_models"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    gliner_model: str = "urchade/gliner_mediumv2.1"
    gliner_threshold: float = 0.4
    enable_llm_entity_validation: bool = False

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
