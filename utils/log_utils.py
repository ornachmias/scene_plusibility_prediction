import json
import logging
import os
import time
from datetime import datetime

default_log_format = '%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'

loggers = {}


def get_logger(logger_name, log_format=None):
    global loggers

    if logger_name not in loggers:
        logging.root.setLevel(logging.DEBUG)

        log_dir = os.path.join('.', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        if log_format is None:
            formatter = logging.Formatter(default_log_format)
        else:
            formatter = logging.Formatter(log_format)

        info_path = os.path.join(log_dir, logger_name + '.log')
        info_handler = logging.FileHandler(info_path)
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)

        debug_handler = logging.StreamHandler()
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)

        logger = logging.getLogger(logger_name)
        logger.addHandler(debug_handler)
        logger.addHandler(info_handler)
        loggers[logger_name] = logger

    return loggers[logger_name]


def log_performance(log: dict):
    log_format = '%(message)s'
    logger = get_logger('performance', log_format)
    log['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    logger.info(json.dumps(log))


def hide_pil_logs():
    logging.getLogger('PIL').setLevel(logging.WARNING)


def hide_matplotlib_logs():
    logging.getLogger('matplotlib.font_manager').disabled = True
