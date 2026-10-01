"""llm_first 链路端到端编排 API。

把 scripts/run_llm_first.py 的编排逻辑(数据加载、记录筛选、上下文构建、
批量消歧/估算、准确率计算)封装为库函数,scripts 退化为薄 CLI。
"""

from __future__ import annotations

import logging
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nallm.core.config import EMBEDDING_DB_DIR, RESULTS_DIR
from nallm.core.similarity import set_org_embedding_getter
from nallm.data import load_data
from nallm.disambiguation import (
    compute_accuracy,
    create_org_embedding_getter,
)
from nallm.disambiguation.batch import (
    ERROR_SENTINEL,
    DisambiguationContext,
    ProgressSnapshot,
    batch_disambiguate,
    batch_estimate_tokens,
    build_ground_truth_index,
)
from nallm.pipelines.io import (
    load_all_records,
    load_processed_records,
    remove_records_from_results,
)

logger = logging.getLogger(__name__)

ORG_DB_PATH = str(EMBEDDING_DB_DIR / "org" / "org.lmdb")


@dataclass
class LlmFirstConfig:
    """llm_first 消歧运行配置。

    LLM 超时/重试不在本配置内——由 openai SDK 默认策略处理
    (timeout=600s, max_retries=2 指数退避, 429 遵从 Retry-After)。
    """

    max_candidates: int = 10
    concurrency: int | None = None  # None → 8
    is_test: bool = False
    save_interval: int = 20


@dataclass
class LlmFirstRun:
    """一次 llm_first 运行的结果汇总。"""

    results: list[dict]
    accuracy: tuple[int, int, float] | None  # (correct, total, accuracy); test/estimate 为 None


@dataclass
class RecordFilter:
    """记录筛选条件(对应 CLI 的 subtype/limit/subtypes_limit/record/overwrite/full)。"""

    subtype: str | None = None
    limit: int | None = None
    subtypes_limit: dict[str, int | None] | None = None
    record: str | None = None
    record_ids: set[str] | None = None
    shard: tuple[int, int] | None = None  # (k, n): crc32(uid) % n == k
    overwrite: bool = False
    full: bool = False


def build_context(
    org_db_path: str = ORG_DB_PATH,
    sim_csv_path: Path | None = None,
    record_ids: list[str] | None = None,
) -> DisambiguationContext:
    """构造 DisambiguationContext 并注册全局 org embedding getter。

    注意:set_org_embedding_getter(core.similarity)是模块级全局副作用,
    必须在引擎跑前调用一次;本函数封装了它。调用方负责 ctx.close()(用 with)。
    """
    ctx = DisambiguationContext.create(org_db_path, sim_csv_path, record_ids)
    set_org_embedding_getter(create_org_embedding_getter(ctx.org_db))
    return ctx


def apply_record_filter(
    records: list[tuple[str, list[dict], str, dict]],
    *,
    filt: RecordFilter | None = None,
    results_path: Path | None = None,
) -> list[tuple[str, list[dict], str, dict]]:
    """按筛选条件过滤记录。

    对应 CLI 的 overwrite / subtype / subtypes_limit / record / full-resume / limit。
    """
    if filt is None:
        return list(records)

    # 覆盖模式：从已有结果移除需重新处理的记录
    if filt.overwrite and filt.full and results_path:
        to_remove: set[str] = set()
        if filt.record:
            to_remove = {filt.record}
        elif filt.subtype:
            to_remove = {r[0] for r in records}
        if to_remove:
            removed = remove_records_from_results(results_path, to_remove)
            if removed > 0:
                logger.info(f"覆盖模式：已移除 {removed} 条记录，将重新处理")

    # 指定记录
    if filt.record:
        records = [r for r in records if r[0] == filt.record]

    # 指定记录 ID 集合(补跑场景:只跑目标集合中缺失的记录)
    if filt.record_ids:
        records = [r for r in records if r[0] in filt.record_ids]
        logger.info(f"ID 集合过滤: 剩余 {len(records)} 条")

    # 分片:按记录 ID 稳定哈希取模,双 key 双进程各跑一片互不重叠(须在 full-resume 前)
    if filt.shard:
        k, n = filt.shard
        records = [r for r in records if zlib.crc32(r[0].encode()) % n == k]
        logger.info(f"分片 {k}/{n}: {len(records)} 条")

    # 指定子类型
    if filt.subtype:
        records = [r for r in records if r[2] == filt.subtype]
        logger.info(f"只处理 {filt.subtype}: {len(records)} 条")

    # 多子类型条目数
    if filt.subtypes_limit:
        filtered_records = []
        for st, lim in filt.subtypes_limit.items():
            st_records = [r for r in records if r[2] == st]
            if lim is not None and lim < len(st_records):
                st_records = st_records[:lim]
            filtered_records.extend(st_records)
            logger.info(f"子类型 {st}: 取 {len(st_records)} 条")
        records = filtered_records
        logger.info(f"多子类型筛选后总计: {len(records)} 条")

    # Full 模式断点续传
    if filt.full and results_path:
        processed = load_processed_records(results_path)
        if processed:
            records = [r for r in records if r[0] not in processed]
            logger.info(f"Full 模式：已处理 {len(processed)} 条，剩余 {len(records)} 条")

    # 限制数量
    if filt.limit is not None and filt.limit < len(records):
        records = records[: filt.limit]
        logger.info(f"限制处理: {filt.limit} 条")

    return records


async def run_llm_first(
    records: list[tuple[str, list[dict], str, dict]],
    data: dict,
    context: DisambiguationContext,
    ground_truth_index: dict[str, str],
    config: LlmFirstConfig | None = None,
    *,
    on_save: Callable[[list[dict]], None] | None = None,
    on_progress: Callable[[ProgressSnapshot], None] | None = None,
) -> LlmFirstRun:
    """核心引擎入口(已加载 records + 已构造 context)。不碰路径、不开 DB。

    调用方需先 set_org_embedding_getter()、构造 context。
    """
    config = config or LlmFirstConfig()
    concurrency = config.concurrency or 8
    logger.info(
        f"Worker 数: {concurrency}"
    )

    results = await batch_disambiguate(
        records,
        data,
        ground_truth_index,
        context,
        max_candidates=config.max_candidates,
        concurrency=concurrency,
        save_interval=config.save_interval,
        is_test=config.is_test,
        on_save=on_save,
        on_progress=on_progress,
    )

    accuracy = None
    if not config.is_test:
        valid_results = [
            r for r in results if r.get("predicted_author_id") != ERROR_SENTINEL
        ]
        accuracy = compute_accuracy(valid_results)

    return LlmFirstRun(results=results, accuracy=accuracy)


async def run_llm_first_from_path(
    split: str = "valid",
    *,
    estimate: bool = False,
    filtered_candidates_path: Path | None = None,
    sim_csv_path: Path | None = None,
    org_db_path: str | None = None,
    results_path: Path | None = None,
    config: LlmFirstConfig | None = None,
    filt: RecordFilter | None = None,
    on_save: Callable[[list[dict]], None] | None = None,
    on_progress: Callable[[ProgressSnapshot], None] | None = None,
) -> LlmFirstRun:
    """端到端:加载 → 筛选 → 构建上下文 → 批量消歧/估算 → 准确率。

    Args:
        split: 数据集 (valid/test)
        estimate: True 则只估算 token 不调 LLM
        filtered_candidates_path: 默认 RESULTS_DIR/{split}_filtered_candidates.json
        sim_csv_path: 默认 RESULTS_DIR/{split}_unass_sim_full.csv
        org_db_path: 默认 org embedding DB
        results_path: full 模式增量保存 + 断点续传路径(为 None 则不增量/续传)
        config: 运行配置
        filt: 记录筛选条件
        on_save: 增量保存回调
        on_progress: 进度回调
    """
    config = config or LlmFirstConfig()
    is_test = split == "test" or config.is_test
    config.is_test = is_test

    filtered_candidates_path = filtered_candidates_path or (
        RESULTS_DIR / f"{split}_filtered_candidates.json"
    )
    sim_csv_path = sim_csv_path or (RESULTS_DIR / f"{split}_unass_sim_full.csv")
    org_db_path = org_db_path or ORG_DB_PATH

    if not filtered_candidates_path.exists():
        raise FileNotFoundError(
            f"文件不存在: {filtered_candidates_path};请先运行 filter_candidates.py"
        )

    logger.info("加载数据...")
    data = load_data()
    ground_truth_index = build_ground_truth_index(data)
    logger.info(f"构建 ground_truth 索引: {len(ground_truth_index)} 条记录")

    records = load_all_records(filtered_candidates_path)
    subtype_counts = Counter(s for _, _, s, _ in records)
    logger.info(f"总记录数: {len(records)}")
    logger.info("子类型分布:")
    for st in sorted(subtype_counts):
        cnt = subtype_counts[st]
        logger.info(f"  {st}: {cnt} ({cnt / len(records) * 100 if records else 0:.1f}%)")

    records = apply_record_filter(records, filt=filt, results_path=results_path)
    if not records:
        logger.warning("没有记录，退出")
        return LlmFirstRun(results=[], accuracy=None)

    record_ids = [r[0] for r in records if r[2] != "NONE"]
    if record_ids:
        logger.info(f"加载相似度数据（{len(record_ids)} 条有候选的记录）...")
        sim_csv = sim_csv_path
    else:
        logger.info("没有有候选的记录，跳过相似度数据加载")
        sim_csv = None

    with build_context(org_db_path, sim_csv, record_ids or None) as ctx:
        if estimate:
            results = batch_estimate_tokens(
                records,
                data,
                ctx,
                max_candidates=config.max_candidates,
                save_interval=config.save_interval,
                is_test=is_test,
                on_save=on_save,
            )
            return LlmFirstRun(results=results, accuracy=None)

        return await run_llm_first(
            records,
            data,
            ctx,
            ground_truth_index,
            config,
            on_save=on_save,
            on_progress=on_progress,
        )
