import logging
import argparse
import os

from datetime import datetime
import sys


class MainLogger:  # Singleton
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 args: argparse.Namespace):
        if self._initialized: return

        self.logger_name = 'main'
        # self.parallel = args.parallel

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if args.log_file == 1:
            log_save_path = "./log"
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)

            datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatter_file = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
            handler_file = logging.FileHandler(os.path.join(log_save_path, f'{datetime_now}.log'))
            handler_file.setLevel(logging.DEBUG)
            handler_file.setFormatter(formatter_file)
            self.logger.addHandler(handler_file)

        self._initialized = True

        def catch_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            logger = logging.getLogger("main")

            logger.error(
                "Unexpected exception.",
                exc_info=(exc_type, exc_value, exc_traceback)
            )

        sys.excepthook = catch_exception

    def __check_gpu_rank(self, gpu_rank: int) -> bool:
        return True
        if self.parallel == 0:
            return True

        if gpu_rank == 0:
            return True
        return False

    def debug(self,
              msg: str,
              gpu_rank: int = -1):
        if not self.__check_gpu_rank(gpu_rank):
            return
        self.logger.debug(msg)

    def info(self,
              msg: str,
              gpu_rank: int = -1):
        if not self.__check_gpu_rank(gpu_rank):
            return
        self.logger.info(msg)

    def warning(self,
              msg: str,
              gpu_rank: int = -1):
        if not self.__check_gpu_rank(gpu_rank):
            return
        self.logger.warning(msg)

    def error(self,
              msg: str,
              gpu_rank: int = -1):
        if not self.__check_gpu_rank(gpu_rank):
            return
        self.logger.error(msg)

    def exception(self,
              msg: str,
              gpu_rank: int = -1):
        if not self.__check_gpu_rank(gpu_rank):
            return
        self.logger.exception(msg)
