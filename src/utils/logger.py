import os
import logging


def get_logger(name, level="warn"):
    """Tạo / lấy logger để in log ra terminal.

    Parameters:
    name : str
        Tên logger.
    level : {'debug', 'info', 'warn', 'warning', 'error', 'critical'}
        Mức log mặc định. Nếu có biến môi trường LOG_LEVEL thì ưu tiên LOG_LEVEL."""
    logger = logging.getLogger(name)

    # Ưu tiên biến môi trường nếu có
    level = os.environ.get("LOG_LEVEL", level)
    level = str(level).lower()

    msg_formats = {
        "debug": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
        "info": "%(asctime)s %(message)s  [at %(filename)s:%(lineno)d]",
        "warn": "%(asctime)s %(message)s",
        "warning": "%(asctime)s %(message)s",
        "error": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
        "critical": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
    }

    level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    # Nếu level không hợp lệ thì fallback về warn
    if level not in msg_formats:
        level = "warn"

    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=msg_formats[level], datefmt=date_format)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Xóa StreamHandler cũ để tránh bị in log lặp
    for h in list(logger.handlers):
        if isinstance(h, logging.StreamHandler):
            logger.removeHandler(h)

    logger.addHandler(handler)
    logger.setLevel(level_mapping[level])

    # Tránh propagate lên root logger gây in trùng
    logger.propagate = False

    return logger