"""
Forward library log records into ROS logging.

Modules under 'logic' use standard :mod:`logging` so they stay importable
without ROS. A node calls :func:`install` once and their records then reach
'/rosout' with real severities.
"""

import logging

#: Root logger for this package. Every 'logic' module is a child of it.
PACKAGE_LOGGER = 'speech_utilities2'


class RosLogHandler(logging.Handler):
    """Re-emit :mod:`logging` records through a ROS logger."""

    def __init__(self, ros_logger):
        """Wrap 'ros_logger', usually 'node.get_logger()'."""
        super().__init__()
        self._ros_logger = ros_logger

    def emit(self, record: logging.LogRecord) -> None:
        """Re-emit one record at the matching ROS severity."""
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        if record.levelno >= logging.ERROR:
            self._ros_logger.error(message)
        elif record.levelno >= logging.WARNING:
            self._ros_logger.warn(message)
        elif record.levelno >= logging.INFO:
            self._ros_logger.info(message)
        else:
            self._ros_logger.debug(message)


def install(ros_logger, level: int = logging.DEBUG) -> None:
    """
    Route this package's library logs to 'ros_logger'.

    Calling this twice replaces the handler rather than adding a second one. The
    default level forwards everything and lets the ROS logger filter, so debug
    records such as prompt dumps stay hidden unless the node runs with
    '--ros-args --log-level debug'.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        if isinstance(handler, RosLogHandler):
            logger.removeHandler(handler)
    logger.addHandler(RosLogHandler(ros_logger))
    logger.setLevel(level)
    # Keep records out of the root logger, which would also print them.
    logger.propagate = False
