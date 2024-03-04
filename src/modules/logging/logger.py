from epochalyst._core._logging._logger import _Logger
from src.logging_utils.logger import logger
from typing import Any
import wandb


class Logger(_Logger):
    """A logger that logs to the terminal and to W&B.
    
    To use this logger, inherit, this will make the following methods available:
    - log_to_terminal(message: str) -> None
    - log_to_debug(message: str) -> None
    - log_to_warning(message: str) -> None
    - log_to_external(message: dict[str, Any], **kwargs: Any) -> None
    - external_define_metric(metric: str, metric_type: str) -> None
    """

    def log_to_terminal(self, message: str) -> None:
        """Log a message to the terminal."""
        logger.info(message)

    def log_to_debug(self, message: str) -> None:
        """Log a message to the debug level."""
        logger.debug(message)

    def log_to_warning(self, message: str) -> None:
        """Log a message to the warning level."""
        logger.warning(message)

    def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None:
        if wandb.run:
            wandb.log(message, **kwargs)

    def external_define_metric(self, metric: str, metric_type: str) -> None:
        if wandb.run:
            wandb.define_metric(metric, summary=metric_type)