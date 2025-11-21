import logging
import os
from pydantic import BaseModel, Field, ValidationError

from .loader import load_environment

logger = logging.getLogger(__name__)

_LOGGING_CONFIGURED = False


class AppSettings(BaseModel):
    environment: str = Field(..., description="Environment name: dev, prod, staging")
    debug: bool = Field(..., description="Enable or disable debug mode")
    openai_api_key: str = Field(..., description="Key for accessing OpenAI API")
    database_url: str = Field(..., description="Database connection string")
    log_level: str = Field(
        "INFO",
        description="Logging level: DEBUG, INFO, WARN, ERROR",
    )


def _configure_logging(level_name: str) -> None:
    """Configure global logging based on level name from settings.

    This is called once, the first time settings are loaded.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        logger.debug("Logging already configured; skipping reconfiguration")
        return

    level_name = (level_name or "INFO").upper()
    # Map 'WARN' to 'WARNING' for logging module compatibility
    if level_name == "WARN":
        level_name = "WARNING"

    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Reduce noise from third-party libraries unless debugging
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("tavily").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.INFO)
        logging.getLogger("langgraph").setLevel(logging.INFO)

    _LOGGING_CONFIGURED = True
    logger.info("Logging configured with level: %s", level_name)


def get_settings() -> AppSettings:
    """Load environment variables and validate them using Pydantic.

    Ensures the application always has correct configuration and initializes logging.
    """
    # Load from .env if not already loaded
    load_environment()

    try:
        # Provide sensible defaults if env vars are missing
        environment = os.getenv("ENVIRONMENT") or "dev"
        debug_env = os.getenv("DEBUG", "true").lower()
        debug = debug_env in {"1", "true", "yes", "y"}
        openai_api_key = os.getenv("OPENAI_API_KEY") or "missing-openai-key"
        database_url = os.getenv("DATABASE_URL") or "sqlite:///./app.db"
        log_level = os.getenv("LOG_LEVEL") or "INFO"

        settings = AppSettings(
            environment=environment,
            debug=debug,
            openai_api_key=openai_api_key,
            database_url=database_url,
            log_level=log_level,
        )

        _configure_logging(settings.log_level)
        logger.debug(
            "Settings loaded: environment=%s, debug=%s, database_url=%s, log_level=%s",
            settings.environment,
            settings.debug,
            settings.database_url,
            settings.log_level,
        )
        return settings
    except ValidationError as e:
        logger.exception("Invalid configuration encountered")
        raise RuntimeError(f"Invalid configuration: {e}")
