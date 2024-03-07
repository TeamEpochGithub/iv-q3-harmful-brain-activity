from src.modules.logging.logger import Logger
from epochalyst.pipeline.model.training.training import TrainingPipeline


class VerboseTrainingPipeline(TrainingPipeline, Logger):
    """A verbose training pipeline that logs to the terminal and to W&B."""