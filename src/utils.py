import logging
from pathlib import Path

def setup_logger(log_path: Path = None, name: str = 'logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if log_path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger