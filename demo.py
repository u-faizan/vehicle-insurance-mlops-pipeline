# below code is to check the logging config
from src.logger import logging
from src.pipline.training_pipeline import TrainPipeline
# from src.exception import MyException
import sys

# logging.debug("This is a debug message.")
# logging.info("This is an info message.")
# logging.warning("This is a warning message.")
# logging.error("This is an error message.")
# logging.critical("This is a critical message.")

# below code is to check the exception config
# try:
#     result = 1 / 0
# except Exception as e:
#     raise MyException(e, sys)

from src.pipline.training_pipeline import TrainPipeline

pipline = TrainPipeline()
pipline.run_pipeline()