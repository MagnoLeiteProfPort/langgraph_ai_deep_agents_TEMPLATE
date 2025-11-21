import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_environment() -> None:
    """Load environment variables from the project's root .env file.

    This function is safe, idempotent, and avoids hardcoding absolute paths.
    """
    current_path = Path(__file__).resolve()
    # .../project_root/src/deep_agents_libs/config/loader.py
    # parents:
    #   [0] = .../project_root/src/deep_agents_libs/config
    #   [1] = .../project_root/src/deep_agents_libs
    #   [2] = .../project_root/src
    #   [3] = .../project_root   <-- we want this
    project_root = current_path.parents[3]
    env_path = project_root / ".env"

    logger.debug("Resolved project root to: %s", project_root)
    logger.debug("Looking for .env file at: %s", env_path)

    if not env_path.exists():
        logger.error(".env file not found at: %s", env_path)
        raise FileNotFoundError(f".env file not found at: {env_path}")

    load_dotenv(dotenv_path=env_path, override=True)
    logger.info(".env file loaded from: %s", env_path)
