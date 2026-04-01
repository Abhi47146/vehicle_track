import logging
from logging.handlers import TimedRotatingFileHandler
import os

class Logger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Log Format: [Timestamp] [Level] [Component] Message
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # 1. Timed Rotating File Handler
        # 'when="D"' means daily rotation
        # 'interval=1' means every 1 day
        # 'backupCount=5' keeps exactly 5 old log files
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, "nayanam.log"),
            when="D",
            interval=1,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        # 2. Console Handler (so you can still see logs in terminal)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Avoid adding multiple handlers if the logger already exists
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, msg): self.logger.info(msg)
    def error(self, msg): self.logger.error(msg)
    def debug(self, msg): self.logger.debug(msg)
    def warning(self, msg): self.logger.warning(msg)