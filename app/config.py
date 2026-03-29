from pydantic_settings import BaseSettings


class EngineConfig(BaseSettings):
    # Runtime
    ollama_host: str = "http://localhost:11434"
    default_runtime: str = "ollama"

    # Context defaults
    default_context_budget: int = 32768
    context_safety_factor: float = 0.7
    default_max_tokens: int = 4096

    # Retry
    max_retries: int = 3
    retry_temperature_decay: float = 0.5

    # Timeouts (seconds)
    llm_connect_timeout: int = 60
    llm_read_timeout: int = 600

    # Database
    database_url: str = "postgresql://sanctum:sanctum@localhost:5432/sanctum_engine"

    # Logging
    log_level: str = "INFO"
    log_task_prompts: bool = False
    log_task_responses: bool = False

    # Model defaults
    default_temperature: float = 0.3
    embedding_model: str = "nomic-embed-text:latest"

    model_config = {"env_prefix": "SANCTUM_"}


settings = EngineConfig()
