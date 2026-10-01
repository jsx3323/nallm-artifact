"""进度渲染与增量保存回调(上层实现,注入引擎)。

引擎(batch)通过 on_save / on_progress 回调槽与上层解耦:
- on_save: 引擎攒够一批成功结果后调用,由本模块的 make_append_store 落盘;
- on_progress: 引擎每秒组装 ProgressSnapshot 调用,由本模块的 render_ansi_status 渲染。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from nallm.disambiguation.batch import ProgressSnapshot, _format_time
from nallm.pipelines.io import append_results


def render_ansi_status(snapshot: ProgressSnapshot) -> None:
    """把进度快照渲染成 ANSI 状态面板(原 batch_disambiguate._display_status 的渲染逻辑)。"""
    # 清屏并移动光标到顶部
    print("\033[2J\033[H", end="")

    total = snapshot.total
    done = snapshot.done
    elapsed = snapshot.elapsed

    # 预估剩余时间
    if done > 0:
        avg_time = elapsed / done
        eta_seconds = avg_time * (total - done)
        eta_str = _format_time(eta_seconds)
    else:
        eta_str = "计算中..."

    elapsed_str = _format_time(elapsed)

    # 标题
    print(f"{'=' * 60}")
    print(f"消歧进度: {done}/{total} | 成功: {snapshot.success} | 失败: {snapshot.error}")
    print(f"{'=' * 60}")

    # 时间信息
    print(f"已用时间: {elapsed_str} | 预估剩余: {eta_str}")

    # 当前运行中的任务
    running_count = len(snapshot.running)
    print(f"\n正在处理 ({running_count} 个并发):\n")

    if snapshot.running:
        # 按开始时间排序
        sorted_running = sorted(
            snapshot.running.items(),
            key=lambda x: x[1][1],
            reverse=True,
        )
        for record_id, (subtype, task_start_time) in sorted_running[
            : snapshot.concurrency
        ]:
            task_elapsed = time.time() - task_start_time
            short_id = record_id.split("-")[0][:8]
            print(f"{short_id} | {subtype:15} | {task_elapsed:5.1f}s")
    else:
        print("(无)")

    print(f"\n{'=' * 60}")
    print("按 Ctrl+C 中断...")


def make_append_store(results_path: Path) -> Callable[[list[dict]], None]:
    """构造增量保存回调(on_save):封装 append_results 到固定路径。

    Args:
        results_path: 结果落盘路径

    Returns:
        on_save 回调,接收一批结果 dict 列表,追加写入 results_path。
    """
    def _save(results: list[dict]) -> None:
        append_results(results, results_path)

    return _save
