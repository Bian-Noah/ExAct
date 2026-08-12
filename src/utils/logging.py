"""日志工具模块。

提供统一的 logger 配置，支持幂等调用（相同 name 不重复添加 handler）。
"""

import logging
import sys


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """配置并返回 logger。

    重复调用相同 name 返回同一实例，不重复加 handler。

    Args:
        name: logger 名称，建议用 __name__ 或模块名。
        level: logging 级别（logging.INFO / logging.DEBUG 等）。

    Returns:
        配置好的 logging.Logger。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 幂等：已有 handler 则不再添加
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
