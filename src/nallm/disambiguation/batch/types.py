"""引擎与上层之间的进度快照类型(放下层,避免反向依赖)。

引擎(batch)通过 ProgressSnapshot 把运行态快照交给上层(pipelines)渲染,
从而引擎无需 import 上层、也无需直接 print。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProgressSnapshot:
    """批量消歧的进度快照,由引擎组装、由上层渲染器消费。

    Attributes:
        total: 记录总数
        done: 已完成数(success + error)
        success: 成功数
        error: 失败数
        elapsed: 已用秒数
        concurrency: 并发 worker 数
        running: 正在处理的记录 {record_id: (subtype, start_time)}
    """

    total: int
    done: int
    success: int
    error: int
    elapsed: float
    concurrency: int
    running: dict[str, tuple[str, float]]
