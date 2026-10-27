import logging
import logging.config
import warnings

import yaml

from src.core.settings import get_settings

settings = get_settings()


def setup_logging() -> None:

    with settings.logging_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    logging.config.dictConfig(config)
    logging.captureWarnings(True)
    warnings.simplefilter("default", DeprecationWarning)
