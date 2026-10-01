"""计时器工具"""

from datetime import datetime


class Timer:
    """简单的计时器类

    用法:
        timer = Timer()
        timer.start()
        # ... 执行操作 ...
        timer.stop()
        print(f"耗时: {timer.elapsed()}")
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """开始计时"""
        self.start_time = datetime.now()

    def stop(self):
        """停止计时"""
        self.end_time = datetime.now()

    def elapsed(self):
        """获取耗时"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        else:
            return None


__all__ = ["Timer"]
