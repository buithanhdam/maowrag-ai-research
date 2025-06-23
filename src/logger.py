# This file contains the logger configuration for the application.
# src/logger.py
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import sys
import click
import logging
from copy import copy
from typing import Literal
import os

TRACE_LOG_LEVEL = 5


class ColourizedFormatter(logging.Formatter):
    level_colors = {
        TRACE_LOG_LEVEL: "blue",
        logging.DEBUG: "cyan",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "magenta",
    }

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "%",
        use_colors: bool | None = None,
    ):
        """
        Initialize the formatter.

        Args:
            fmt (str | None): The format string. Defaults to None.
            datefmt (str | None): The date format string. Defaults to None.
            style (Literal["%", "{", "$"]): The style of the format string. Defaults to %.
            use_colors (bool | None): Whether to use colors. Defaults to None.
        """
        if use_colors in (True, False):
            self.use_colors = use_colors
        else:
            self.use_colors = sys.stdout.isatty()
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

    def color_level_name(self, level_name: str, level_no: int) -> str:
        """
        Colorize the level name.

        Args:
            level_name (str): The level name.
            level_no (int): The level number.

        Returns:
            str: The colorized level name.
        """
        color = self.level_colors.get(level_no, "reset")
        return click.style(str(level_name), fg=color)

    def color_message(self, message: str, level_no: int) -> str:
        """
        Colorize the message.

        Args:
            message (str): The message.
            level_no (int): The level number.

        Returns:
            str: The colorized message.
        """
        color = self.level_colors.get(level_no, "reset")
        return click.style(str(message), fg=color)

    def color_date(self, record: logging.LogRecord) -> str:
        """
        Apply green color to the date.

        Args:
            record (logging.LogRecord): The log record.

        Returns:
            str: The colorized date.
        """
        date_str = self.formatTime(record, self.datefmt)
        return click.style(date_str, fg=(200, 200, 200))

    def should_use_colors(self) -> bool:
        """
        Check if colors should be used. Defaults to True.

        Returns:
            bool: Whether colors should be used.
        """
        return self.use_colors

    def formatMessage(self, record: logging.LogRecord) -> str:
        """
        Format the message.

        Args:
            record (logging.LogRecord): The log record.

        Returns:
            str: The formatted message.
        """
        recordcopy = copy(record)
        levelname = recordcopy.levelname
        seperator = " " * (8 - len(recordcopy.levelname))

        if self.use_colors:
            relpathname = "/".join(recordcopy.pathname.split("/")[-2:])
            levelname = self.color_level_name(levelname, recordcopy.levelno)
            recordcopy.msg = self.color_message(recordcopy.msg, recordcopy.levelno)
            recordcopy.__dict__["message"] = recordcopy.getMessage()
            recordcopy.asctime = self.color_date(recordcopy)
            recordcopy.__dict__["relpathname"] = relpathname

        recordcopy.__dict__["levelprefix"] = levelname + seperator
        return super().formatMessage(recordcopy)


class DefaultFormatter(ColourizedFormatter):
    def should_use_colors(self) -> bool:
        return sys.stderr.isatty()

    def formatMessage(self, record: logging.LogRecord) -> str:
        recordcopy = copy(record)

        if "pathname" in recordcopy.__dict__:
            relpathname = "/".join(recordcopy.pathname.split("/")[-2:])
            recordcopy.__dict__["relpathname"] = relpathname
        else:
            recordcopy.__dict__["relpathname"] = (
                "N/A"  # Fallback when pathname is missing
            )

        levelname = recordcopy.levelname
        seperator = " " * (8 - len(levelname))

        if self.use_colors:
            levelname = self.color_level_name(levelname, recordcopy.levelno)
            recordcopy.msg = self.color_message(recordcopy.msg, recordcopy.levelno)
            recordcopy.__dict__["message"] = recordcopy.getMessage()
            recordcopy.asctime = self.color_date(recordcopy)

        recordcopy.__dict__["levelprefix"] = levelname + seperator
        return super().formatMessage(recordcopy)


class FileFormater(logging.Formatter):
    def formatMessage(self, record: logging.LogRecord) -> str:
        recordcopy = copy(record)

        if "pathname" in recordcopy.__dict__:
            relpathname = "/".join(recordcopy.pathname.split("/")[-2:])
            recordcopy.__dict__["relpathname"] = relpathname
        else:
            recordcopy.__dict__["relpathname"] = (
                "N/A"  # Fallback when pathname is missing
            )

        return super().formatMessage(recordcopy)


class SafeDailyFolderFileHandler(TimedRotatingFileHandler):
    """A safe handler that creates logs in daily folders with proper permissions"""

    def __init__(
        self,
        filename,
        when="D",
        interval=1,
        backupCount=0,
        encoding=None,
        delay=False,
        utc=True,
        atTime=None,
    ):
        # Create base logs directory with proper permissions
        self.base_log_dir = "/app/logs"
        self._ensure_log_directory()

        # Use today's date for folder structure
        today = datetime.now().strftime("%Y/%m/%d")
        folder_path = os.path.join(self.base_log_dir, today)
        self._ensure_directory(folder_path)

        file_path = os.path.join(folder_path, filename)
        super().__init__(
            file_path,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            utc=utc,
            atTime=atTime,
        )

        # Store the base pattern for the filename
        self.base_filename_pattern = filename

    def _ensure_log_directory(self):
        """Ensure base log directory exists with proper permissions"""
        try:
            os.makedirs(self.base_log_dir, mode=0o777, exist_ok=True)
            # Try to set permissions (may fail in some environments)
            try:
                os.chmod(self.base_log_dir, 0o777)
            except (OSError, PermissionError):
                pass  # Ignore permission errors
        except Exception as e:
            # Fallback to /tmp if we can't create in /app/logs
            self.base_log_dir = "/tmp/app_logs"
            os.makedirs(self.base_log_dir, mode=0o777, exist_ok=True)

    def _ensure_directory(self, path):
        """Ensure directory exists with proper permissions"""
        try:
            os.makedirs(path, mode=0o777, exist_ok=True)
            try:
                os.chmod(path, 0o777)
            except (OSError, PermissionError):
                pass
        except Exception:
            pass  # Continue even if directory creation fails

    def _update_filename(self):
        """Update the filename based on current date"""
        today = datetime.now().strftime("%Y/%m/%d")
        folder_path = os.path.join(self.base_log_dir, today)
        self._ensure_directory(folder_path)

        # Get the base filename (without path)
        base_name = os.path.basename(self.base_filename_pattern)

        # Set the new filename with updated path
        self.baseFilename = os.path.join(folder_path, base_name)

    def emit(self, record):
        """Check if date has changed before emitting the record"""
        try:
            current_date = datetime.now().strftime("%Y/%m/%d")
            expected_path = os.path.join(self.base_log_dir, current_date)
            current_dir = os.path.dirname(self.baseFilename)

            # If date has changed, update the filename
            if current_dir != expected_path:
                self.close()  # Close current file
                self._update_filename()  # Update filename with new date

                # Create the file if it doesn't exist
                if not self.delay:
                    self.stream = self._open()

            super().emit(record)
        except Exception as e:
            # If file logging fails, just continue without crashing
            self.handleError(record)

_loggers = {}


def get_formatted_logger(
    name: str,
    file_name: str | None = None,
) -> logging.Logger:
    """
    Get a coloured logger with optional Seq integration.

    Args:
        name (str): The name of the logger.
        file_name (str | None): The path name to the log file. Defaults to None.

    Returns:
        logging.Logger: The logger object.

    **Note:** Name is used as an identifier to prevent duplicate loggers and for hierarchical logging.
    """

    file_name = file_name or "backend.log"

    # Return existing logger if already created
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name=name)
    logger.setLevel(TRACE_LOG_LEVEL)

    # Only add handlers if none exist
    if not logger.hasHandlers():
        # Console handler
        stream_handler = logging.StreamHandler()
        stream_formatter = DefaultFormatter(
            "%(asctime)s | %(levelprefix)s - [%(relpathname)s %(funcName)s(%(lineno)d)] - %(message)s",
            datefmt="%Y/%m/%d  %H:%M:%S",
        )
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # File handler (with fallback for permission issues)
        try:
            file_handler = SafeDailyFolderFileHandler(
                file_name, when="midnight", interval=1, encoding="utf-8"
            )
            file_formatter = FileFormater(
                "%(asctime)s | %(levelname)-8s - [%(relpathname)s %(funcName)s(%(lineno)d)] - %(message)s",
                datefmt="%Y/%m/%d - %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # If file handler fails, log to console only
            logger.warning(f"Could not create file handler: {e}")

        # Store logger in cache
        _loggers[name] = logger
    return logger