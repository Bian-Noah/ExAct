"""utils.logging 单元测试。"""

import logging

from utils.logging import setup_logging


def test_setup_logging_returns_logger():
    """返回值为 Logger 实例，name 匹配。"""
    logger = setup_logging("test_logger_a")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger_a"


def test_setup_logging_sets_level():
    """INFO / DEBUG 级别设置正确。"""
    logger_debug = setup_logging("test_level_debug", level=logging.DEBUG)
    assert logger_debug.level == logging.DEBUG

    logger_info = setup_logging("test_level_info_default")
    assert logger_info.level == logging.INFO


def test_setup_logging_has_stream_handler():
    """至少 1 个 StreamHandler，formatter 包含 asctime/levelname/name。"""
    logger = setup_logging("test_handler")
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) >= 1

    fmt = stream_handlers[0].formatter
    assert fmt is not None
    fmt_str = fmt._fmt
    assert "%(asctime)s" in fmt_str
    assert "%(levelname)s" in fmt_str
    assert "%(name)s" in fmt_str


def test_setup_logging_idempotent_same_name():
    """两次调用后 handler 数量不增加。"""
    logger1 = setup_logging("test_same")
    count1 = len(logger1.handlers)
    logger2 = setup_logging("test_same")
    count2 = len(logger2.handlers)
    assert count1 == count2
    assert logger1 is logger2


def test_setup_logging_outputs_to_stderr(capsys):
    """日志输出到 stderr，内容包含 message/levelname/name。"""
    logger = setup_logging("test_capsys", level=logging.INFO)
    logger.warning("hello world")
    captured = capsys.readouterr()
    assert "hello world" in captured.err
    assert "WARNING" in captured.err
    assert "test_capsys" in captured.err
