# from typing import Tuple
# import datetime
import logging

from src.config import PATH_DATA_LOGS


def get_logger(log_name: str) -> logging.Logger:
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Logger to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Logger to file
    # file_path = log_name + datetime.datetime.today().strftime("_%Y_%m_%d.log")
    file_path = log_name + ".log" # + datetime.datetime.today().strftime("_%Y_%m_%d.log")
    log_file_path = PATH_DATA_LOGS / file_path  # str(log_dir_path / file_path )
    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def logger_args(logger, parser) -> None:
    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        logger.info(" {0}: {1}".format(k, v))
