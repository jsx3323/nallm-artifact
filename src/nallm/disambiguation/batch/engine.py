"""消歧批处理引擎(单条 + 批量 + token 估算)。

下沉自 scripts/run_llm_first.py。纯编排逻辑,通过注入 seam 与上层解耦:
- on_save: 增量落盘回调(替代直写文件)
- on_progress: 进度快照回调(替代直接 ANSI print)

LLM 调用走纯 OpenAI SDK(chat_completion),超时/重试/限流全部由 SDK 默认策略处理。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from tqdm.asyncio import tqdm

from nallm.classification import calibrate_confidence
from nallm.core.similarity import compute_org_similarity
from nallm.data.similarity import aggregate_candidate_stats
from nallm.disambiguation import (
    STRATEGY_PROMPT_TYPE_MAP,
    DisambiguationOutput,
    DisambiguationStrategy,
    build_human_text,
    build_messages,
    chat_completion,
    construct_prompt_data,
    estimate_tokens,
    get_strategy,
    get_system_prompt_by_strategy,
    parse_output,
)
from nallm.disambiguation.batch.context import DisambiguationContext
from nallm.disambiguation.batch.results import (
    ERROR_SENTINEL,
    NONE_AUTHOR_ID,
    _handle_exception,
    auto_assign_strong,
    get_ground_truth_author_id,
    make_result_dict,
)
from nallm.disambiguation.batch.types import ProgressSnapshot

logger = logging.getLogger(__name__)


def estimate_record_tokens(
    unass_record: str,
    candidates: list[dict],
    subtype: str,
    data: dict,
    context: DisambiguationContext,
    max_candidates: int = 10,
    is_test: bool = False,
) -> dict:
    """估算单条记录的 prompt token 数

    复用 disambiguate_record 的 prompt 构造逻辑，
    但跳过 LLM 调用和后续处理。

    Args:
        unass_record: 待消歧记录ID (pid-order)
        candidates: 候选作者列表
        subtype: 子类型
        data: 数据字典
        context: 消歧上下文
        max_candidates: 最大候选数

    Returns:
        估算结果字典
    """
    sim_data = context.sim_data
    strategy = get_strategy(subtype)

    # NONE: 无候选
    if strategy == DisambiguationStrategy.NONE:
        return {
            "unass_record": unass_record,
            "subtype": subtype,
            "strategy": strategy.value,
            "estimated_tokens": 0,
            "num_candidates": 0,
        }

    # AUTO: P1-Strong 自动分配，无 prompt
    if strategy == DisambiguationStrategy.AUTO:
        return {
            "unass_record": unass_record,
            "subtype": subtype,
            "strategy": strategy.value,
            "estimated_tokens": 0,
            "num_candidates": len(candidates),
        }

    # 使用共享函数构造 prompt 数据
    prompt_data = construct_prompt_data(
        unass_record=unass_record,
        candidates=candidates,
        data=data,
        strategy=strategy,
        max_candidates=max_candidates,
        is_test=is_test,
        sim_data=sim_data,
        compute_org_similarity_fn=compute_org_similarity,
        get_system_prompt_fn=get_system_prompt_by_strategy,
        build_human_text_fn=build_human_text,
    )

    if prompt_data is None:
        return {
            "unass_record": unass_record,
            "subtype": subtype,
            "strategy": strategy.value,
            "estimated_tokens": 0,
            "num_candidates": len(candidates),
        }

    estimated_tokens = estimate_tokens(prompt_data.system_prompt) + estimate_tokens(
        prompt_data.human_text
    )

    return {
        "unass_record": unass_record,
        "subtype": subtype,
        "strategy": strategy.value,
        "estimated_tokens": estimated_tokens,
        "num_candidates": len(candidates),
    }


async def disambiguate_record(
    unass_record: str,
    candidates: list[dict],
    subtype: str,
    subtype_details: dict,
    data: dict,
    ground_truth_index: dict[str, str],
    context: DisambiguationContext,
    max_candidates: int = 10,
    is_test: bool = False,
) -> dict:
    """消歧处理单条记录（LLM 调用走纯 OpenAI SDK，超时/重试用 SDK 默认）

    Args:
        unass_record: 待消歧记录ID (pid-order)
        candidates: 候选作者列表
        subtype: 子类型
        subtype_details: 子类型详细信息
        data: 数据字典（用于构造 prompt）
        ground_truth_index: pub_id -> author_id 索引
        context: 消歧上下文
        max_candidates: 最大候选数
        is_test: 是否测试集

    Returns:
        消歧结果字典
    """
    sim_data = context.sim_data

    strategy = get_strategy(subtype)

    # 预先计算 ground_truth（O(1) 查找）
    ground_truth = get_ground_truth_author_id(unass_record, ground_truth_index)

    if strategy == DisambiguationStrategy.NONE:
        return make_result_dict(
            unass_record=unass_record,
            predicted_author_id=NONE_AUTHOR_ID,
            confidence="high",
            reasoning="无候选作者",
            ground_truth=ground_truth,
            num_candidates=0,
            subtype=subtype,
            strategy=strategy.value,
            stats={},
        )

    # 计算统计信息（仅非 NONE 类型需要）
    pub_sims = (
        sim_data.get(unass_record, {}).get(candidates[0]["aid"], {})
        if candidates
        else {}
    )
    stats = aggregate_candidate_stats(pub_sims)

    if strategy == DisambiguationStrategy.AUTO:
        response = auto_assign_strong(candidates)
        return make_result_dict(
            unass_record=unass_record,
            predicted_author_id=response.author_id,
            confidence=response.confidence,
            reasoning=response.reasoning,
            ground_truth=ground_truth,
            num_candidates=len(candidates),
            subtype=subtype,
            strategy=strategy.value,
            stats=stats,
        )

    # 以下为需要 LLM 的策略
    coauthor_count = subtype_details.get("coauthor_count", 0)
    org_verified = subtype_details.get("org_verified", False)

    prompt_data = construct_prompt_data(
        unass_record=unass_record,
        candidates=candidates,
        data=data,
        strategy=strategy,
        max_candidates=max_candidates,
        is_test=is_test,
        sim_data=sim_data,
        compute_org_similarity_fn=compute_org_similarity,
        get_system_prompt_fn=get_system_prompt_by_strategy,
        build_human_text_fn=build_human_text,
    )

    if prompt_data is None:
        raise ValueError(f"无效的记录 ID 格式或数据缺失: {unass_record}")

    estimated_prompt_tokens = estimate_tokens(
        prompt_data.system_prompt
    ) + estimate_tokens(prompt_data.human_text)

    # 调用 LLM（纯 OpenAI SDK，超时/重试为 SDK 默认）
    messages = build_messages(
        STRATEGY_PROMPT_TYPE_MAP.get(strategy, "standard"),
        prompt_data.invoke_params,
    )
    content = await chat_completion(messages)
    response: DisambiguationOutput = parse_output(content)

    calibrated_confidence = calibrate_confidence(
        llm_confidence=response.confidence,
        title_sim=stats["max_title_sim"],
        coauthor_count=coauthor_count,
        org_sim=stats["max_org_sim"],
        venue_sim=stats["max_venue_sim"],
        org_verified=org_verified,
        predicted_author_id=response.author_id,
        subtype=subtype,
    )

    return make_result_dict(
        unass_record=unass_record,
        predicted_author_id=response.author_id,
        confidence=calibrated_confidence,
        reasoning=response.reasoning,
        ground_truth=ground_truth,
        num_candidates=len(candidates),
        subtype=subtype,
        strategy=strategy.value,
        stats=stats,
        estimated_tokens=estimated_prompt_tokens,
    )


def batch_estimate_tokens(
    records: list[tuple[str, list[dict], str, dict]],
    data: dict,
    context: DisambiguationContext,
    max_candidates: int = 10,
    save_interval: int = 20,
    is_test: bool = False,
    on_save: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    """批量估算 prompt token 数

    Args:
        records: [(unass_record, candidates, subtype, subtype_details), ...]
        data: 数据字典
        context: 消歧上下文
        max_candidates: 每条记录最大候选数
        save_interval: 增量保存间隔（每处理多少条触发一次 on_save）
        is_test: 是否是测试集
        on_save: 增量保存回调，接收一批结果 dict；为 None 则不保存

    Returns:
        估算结果列表
    """
    all_results = []

    for record, candidates, subtype, _ in tqdm(
        records, desc="估算 token 数", leave=True
    ):
        result = estimate_record_tokens(
            record,
            candidates,
            subtype,
            data,
            context,
            max_candidates,
            is_test,
        )
        all_results.append(result)

        # 增量保存
        if on_save is not None and len(all_results) % save_interval == 0:
            on_save(all_results)
            all_results = []

    # 最终保存
    if on_save is not None and all_results:
        on_save(all_results)

    return all_results


async def batch_disambiguate(
    records: list[tuple[str, list[dict], str, dict]],
    data: dict,
    ground_truth_index: dict[str, str],
    context: DisambiguationContext,
    max_candidates: int = 10,
    concurrency: int = 10,
    save_interval: int = 10,
    is_test: bool = False,
    on_save: Callable[[list[dict]], None] | None = None,
    on_progress: Callable[[ProgressSnapshot], None] | None = None,
) -> list[dict]:
    """批量处理记录

    Args:
        records: [(unass_record, candidates, subtype, subtype_details), ...]
        data: 数据字典
        ground_truth_index: pub_id -> author_id 索引
        context: 消歧上下文
        max_candidates: 每条记录最大候选数，默认 10
        concurrency: 并发数
        save_interval: 增量保存间隔（每处理多少条触发一次 on_save）
        is_test: 是否是测试集
        on_save: 增量保存回调，接收一批成功结果；为 None 则不保存
        on_progress: 进度快照回调（每秒/状态变化时触发）；为 None 则不显示

    Returns:
        消歧结果列表（含错误结果，用于统计）
    """
    # 任务状态跟踪
    # record_id -> (subtype, start_time)
    running_records: dict[str, tuple[str, float]] = {}
    success_count = 0
    error_count = 0
    lock = asyncio.Lock()

    # 时间统计
    start_time = time.time()

    def _emit_progress() -> None:
        """组装进度快照并交给上层渲染回调"""
        if on_progress is None:
            return
        snapshot = ProgressSnapshot(
            total=len(records),
            done=success_count + error_count,
            success=success_count,
            error=error_count,
            elapsed=time.time() - start_time,
            concurrency=concurrency,
            running=dict(running_records),
        )
        on_progress(snapshot)

    async def refresh_display():
        """定时刷新显示"""
        while True:
            await asyncio.sleep(1.0)
            async with lock:
                _emit_progress()

    async def process_one(record, candidates, subtype, subtype_details) -> dict:
        """处理单条记录（LLM 调用，SDK 默认超时重试）"""
        nonlocal success_count, error_count

        # 标记开始
        async with lock:
            running_records[record] = (subtype, time.time())
            _emit_progress()

        try:
            result = await disambiguate_record(
                record,
                candidates,
                subtype,
                subtype_details,
                data,
                ground_truth_index,
                context,
                max_candidates,
                is_test,
            )
        except Exception as e:
            result = _handle_exception(
                e, record, candidates, subtype, ground_truth_index
            )
            if result is None:  # 致命错误
                raise
        finally:
            # 标记结束
            async with lock:
                if record in running_records:
                    del running_records[record]

        # 更新计数
        async with lock:
            if result.get("predicted_author_id") != ERROR_SENTINEL:
                success_count += 1
            else:
                error_count += 1

        return result

    async def worker(queue: asyncio.Queue):
        """Worker 协程：从队列取任务并处理"""
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            record, candidates, subtype, subtype_details = item
            try:
                result = await process_one(record, candidates, subtype, subtype_details)
                all_results.append(result)
                if result.get("predicted_author_id") != ERROR_SENTINEL:
                    successful_results.append(result)

                # 增量保存
                if (
                    on_save is not None
                    and len(all_results) % save_interval == 0
                    and successful_results
                ):
                    on_save(successful_results)
                    successful_results.clear()
            except Exception as e:
                logger.error(f"Worker 异常: {e}")
            finally:
                queue.task_done()

    # 启动定时刷新任务
    refresh_task = asyncio.create_task(refresh_display()) if on_progress is not None else None

    all_results = []
    successful_results = []

    # 创建任务队列，放入所有记录
    queue: asyncio.Queue = asyncio.Queue()
    for record, candidates, subtype, subtype_details in records:
        queue.put_nowait((record, candidates, subtype, subtype_details))

    # 启动固定数量的 worker
    workers = [asyncio.create_task(worker(queue)) for _ in range(concurrency)]

    # 等待队列处理完毕
    try:
        await queue.join()
    finally:
        # 取消刷新任务
        if refresh_task:
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass
        # 停止 worker
        for _ in range(concurrency):
            await queue.put(None)
        await asyncio.gather(*workers, return_exceptions=True)

    # 最终保存
    if on_save is not None and successful_results:
        on_save(successful_results)

    # 最终状态显示
    if on_progress is not None:
        _emit_progress()

    # 返回所有结果（包括错误，用于统计）
    return all_results
