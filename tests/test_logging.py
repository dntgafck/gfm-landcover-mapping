import logging

from utils.logging import get_logger, setup_logging


def test_logging_setup():
    setup_logging(level=logging.DEBUG)
    logger = get_logger("test_logger")
    assert (
        logger.level == logging.NOTSET
    )  # Logger level is usually NOTSET, it inherits from root
    assert logging.getLogger().level == logging.DEBUG
