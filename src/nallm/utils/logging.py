"""日志配置工具"""

import logging


def suppress_http_logging():
    """配置 HTTP 相关日志级别为 WARNING，避免干扰进度条显示"""
    for logger_name in [
        "openai",
        "urllib3",
        "http.client",
        "httpx",
        "httpcore",
        "requests",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


__all__ = ["suppress_http_logging"]
