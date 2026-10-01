"""
综合消歧模块(llm_first 链路 CLI 入口)

编排逻辑在 nallm.pipelines.llm_first(run_llm_first_from_path 端到端 API)。
本脚本仅保留:CLI 参数解析、运行配置构造、统计打印、结果落盘。

LLM 调用为纯 OpenAI SDK,超时/重试全部走 SDK 默认(timeout=600s,
max_retries=2 指数退避,429 遵从 Retry-After),无 rpm/tpm/timeout 参数。
思考开关等扩展参数经 LLM_EXTRA_BODY 环境变量透传(JSON)。

用法:
    # 处理全部数据
    uv run python scripts/run_llm_first.py --full

    # 处理单个子类型
    uv run python scripts/run_llm_first.py --subtype P1-Medium-Comp --limit 50

    # 处理多个子类型，每个指定条目数
    uv run python scripts/run_llm_first.py --subtypes-limit "P1-Medium-Comp:50,P2:50,P3:50"

    # 估算模式：只计算 prompt token 数，不调用 LLM
    uv run python scripts/run_llm_first.py --estimate --full

    # 思考模式(示例)
    LLM_EXTRA_BODY='{"reasoning_effort": "max"}' uv run python scripts/run_llm_first.py --full

    # 实时显示每个案例的处理情况
    uv run python scripts/run_llm_first.py --verbose --limit 10
"""

import argparse
import asyncio
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from nallm.core.config import LOGS_DIR, RESULTS_DIR
from nallm.disambiguation import DisambiguationStrategy, compute_accuracy
from nallm.disambiguation.batch import ERROR_SENTINEL
from nallm.pipelines.llm_first import (
    LlmFirstConfig,
    RecordFilter,
    run_llm_first_from_path,
)
from nallm.pipelines.progress import make_append_store, render_ansi_status
from nallm.utils import Timer, suppress_http_logging
from nallm.utils.io import safe_write_csv

logger = logging.getLogger(__name__)

load_dotenv()  # 不 override:shell 已设的变量(如按进程注入的 LLM_API_KEY)优先于 .env

OUTPUT_DIR = RESULTS_DIR / "disambiguation"

FULL_RESULTS_PATH = OUTPUT_DIR / "valid_full_results.csv"
TEST_FULL_RESULTS_PATH = OUTPUT_DIR / "test_full_results.csv"

ESTIMATE_RESULTS_PATH = OUTPUT_DIR / "valid_estimate_results.csv"
TEST_ESTIMATE_RESULTS_PATH = OUTPUT_DIR / "test_estimate_results.csv"


class TqdmLoggingHandler(logging.Handler):
    """自定义日志处理器，使用 tqdm.write() 输出避免干扰进度条"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="综合消歧模块(纯 OpenAI SDK,超时/重试为 SDK 默认)")
    parser.add_argument("--limit", "-n", type=int, default=None, help="限制处理的记录数，默认处理全部")
    parser.add_argument("--concurrency", "-c", type=int, default=None, help="并发 worker 数，默认 8")
    parser.add_argument("--subtype", "-s", type=str, default=None, help="只处理指定子类型的记录（如 P1-Strong, P0-Low）")
    parser.add_argument("--subtypes-limit", "-S", type=str, default=None, help="指定多个子类型及其条目数，格式: 'P1-Medium-Comp:50,P2:50,P3:50'")
    parser.add_argument("--full", "-f", action="store_true", help="Full 模式：处理全量数据，启用断点续传和增量保存")
    parser.add_argument("--record", "-r", type=str, default=None, help="测试指定记录 ID（如 '001W9YKh-pid123'），仅处理该记录")
    parser.add_argument("--max-candidates", type=int, default=10, help="每条记录的最大候选作者数，默认: 10")
    parser.add_argument("--dataset", "-d", type=str, choices=["valid", "test"], default="valid", help="数据集选择，默认: valid")
    parser.add_argument("--estimate", action="store_true", help="估算模式：只计算 prompt token 数，不调用 LLM")
    parser.add_argument("--overwrite", action="store_true", help="覆盖模式：覆盖已有结果中需要重新处理的记录")
    parser.add_argument("--shard", type=str, default=None, help="分片 k:n——按记录 ID 稳定哈希取第 k 片（双 key 双进程并行用）")
    parser.add_argument("--output", "-o", type=str, default=None, help="full 模式结果路径覆盖（分片时各写各的文件）")
    parser.add_argument("--verbose", "-v", action="store_true", help="实时显示每个案例的处理情况")
    return parser.parse_args()


def parse_shard(s: str | None) -> tuple[int, int] | None:
    """解析 'k:n' 为 (k, n)，非法即抛。"""
    if not s:
        return None
    k, _, n = s.partition(":")
    k_i, n_i = int(k), int(n)
    if not (0 <= k_i < n_i <= 64):
        raise ValueError(f"非法分片 {s}，要求 0 <= k < n <= 64")
    return k_i, n_i


def parse_subtypes_limit(s: str | None) -> dict[str, int | None] | None:
    """解析 'P1:50,P2:50' 为 {subtype: limit}。"""
    if not s:
        return None
    result: dict[str, int | None] = {}
    for item in s.split(","):
        if ":" in item:
            st, lim = item.split(":", 1)
            result[st.strip()] = int(lim.strip())
        else:
            result[item.strip()] = None
    return result


async def main():
    """主函数"""
    suppress_http_logging()
    args = parse_args()

    # 配置日志
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOGS_DIR / "disambiguation.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("Prompt Token 估算模式" if args.estimate else "综合消歧开始")
    logger.info(f"数据集: {args.dataset}")

    is_test = args.dataset == "test"
    if args.estimate:
        results_path = TEST_ESTIMATE_RESULTS_PATH if is_test else ESTIMATE_RESULTS_PATH
    else:
        results_path = TEST_FULL_RESULTS_PATH if is_test else FULL_RESULTS_PATH
    if args.output:
        results_path = Path(args.output)

    on_save = make_append_store(results_path) if args.full else None
    on_progress = render_ansi_status if args.verbose else None
    config = LlmFirstConfig(
        max_candidates=args.max_candidates,
        concurrency=args.concurrency,
        is_test=is_test,
        save_interval=20,
    )
    filt = RecordFilter(
        subtype=args.subtype,
        limit=args.limit,
        subtypes_limit=parse_subtypes_limit(args.subtypes_limit),
        record=args.record,
        shard=parse_shard(args.shard),
        overwrite=args.overwrite,
        full=args.full,
    )

    timer = Timer()
    timer.start()
    run = await run_llm_first_from_path(
        args.dataset,
        estimate=args.estimate,
        results_path=results_path if args.full else None,
        config=config,
        filt=filt,
        on_save=on_save,
        on_progress=on_progress,
    )
    timer.stop()
    logger.info(f"处理用时: {timer.elapsed()}")

    results = run.results

    if args.estimate:
        # 估算模式统计
        total_tokens = sum(r["estimated_tokens"] for r in results)
        avg_tokens = total_tokens / len(results) if results else 0
        logger.info(f"总 Token 数: {total_tokens:,}")
        logger.info(f"平均 Token 数: {avg_tokens:.2f}")

        logger.info("-" * 40)
        logger.info("按子类型统计:")
        subtype_counts = Counter(r["subtype"] for r in results)
        for subtype in sorted(subtype_counts.keys()):
            count = subtype_counts[subtype]
            subtype_results = [r for r in results if r["subtype"] == subtype]
            avg_s = sum(r["estimated_tokens"] for r in subtype_results) / count
            logger.info(f"  {subtype}: {count} 条, 平均 {avg_s:.0f} tokens")

        logger.info("-" * 40)
        logger.info("按策略统计:")
        strategy_counts = Counter(r["strategy"] for r in results)
        for strategy in sorted(strategy_counts.keys()):
            count = strategy_counts[strategy]
            strategy_results = [r for r in results if r["strategy"] == strategy]
            avg_s = sum(r["estimated_tokens"] for r in strategy_results) / count
            logger.info(f"  {strategy}: {count} 条, 平均 {avg_s:.0f} tokens")

        # 保存结果
        if not args.full:
            results_df = pd.DataFrame(results)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_prefix = "test" if is_test else "valid"
            output_path = OUTPUT_DIR / f"{dataset_prefix}_estimate_{timestamp}.csv"
            safe_write_csv(results_df, output_path)
            logger.info(f"结果已保存到: {output_path}")
        else:
            logger.info(f"结果已增量保存到: {results_path}")

    else:
        # 正常模式统计
        valid_results = [r for r in results if r.get("predicted_author_id") != ERROR_SENTINEL]
        error_count = len(results) - len(valid_results)

        if error_count > 0:
            logger.warning(f"错误记录数: {error_count}")

        # 计算准确率（仅验证集有 ground truth）
        if not is_test:
            correct, total, accuracy = run.accuracy  # type: ignore[misc]
            logger.info(f"总准确率: {correct}/{total} = {accuracy:.2%}")
        else:
            logger.info(f"测试集共处理: {len(valid_results)} 条")

        # 预估 prompt token 统计
        total_estimated_tokens = sum(
            r.get("estimated_prompt_tokens", 0) for r in valid_results
        )
        avg_estimated_tokens = (
            total_estimated_tokens / len(valid_results) if valid_results else 0
        )
        logger.info("预估 Prompt Token:")
        logger.info(f"  总计: {total_estimated_tokens:,}")
        logger.info(f"  平均: {avg_estimated_tokens:.2f}")

        # 按策略统计准确率（仅验证集）
        if not is_test:
            logger.info("-" * 40)
            logger.info("按策略统计准确率:")
            for strategy in DisambiguationStrategy:
                strategy_results = [
                    r for r in valid_results if r.get("strategy") == strategy.value
                ]
                if strategy_results:
                    correct_s, total_s, accuracy_s = compute_accuracy(strategy_results)
                    logger.info(
                        f"  {strategy.value}: {correct_s}/{total_s} = {accuracy_s:.2%}"
                    )

            # 按子类型统计准确率
            logger.info("-" * 40)
            logger.info("按子类型统计准确率:")
            result_subtype_counts = Counter(r["subtype"] for r in valid_results)
            for subtype in sorted(result_subtype_counts.keys()):
                subtype_results = [r for r in valid_results if r["subtype"] == subtype]
                correct_s, total_s, accuracy_s = compute_accuracy(subtype_results)
                logger.info(f"  {subtype}: {correct_s}/{total_s} = {accuracy_s:.2%}")

            # 按置信度统计
            logger.info("-" * 40)
            logger.info("按置信度统计:")
            for conf in ["high", "medium", "low"]:
                conf_results = [r for r in valid_results if r["confidence"] == conf]
                if conf_results:
                    correct_c, total_c, accuracy_c = compute_accuracy(conf_results)
                    logger.info(f"  {conf}: {correct_c}/{total_c} = {accuracy_c:.2%}")
        else:
            # 测试集统计（无 ground truth）
            logger.info("-" * 40)
            logger.info("测试集统计:")
            result_subtype_counts = Counter(r["subtype"] for r in valid_results)
            for subtype in sorted(result_subtype_counts.keys()):
                count = result_subtype_counts[subtype]
                logger.info(f"  {subtype}: {count}")

            # 按置信度统计
            logger.info("-" * 40)
            logger.info("按置信度统计:")
            for conf in ["high", "medium", "low"]:
                conf_results = [r for r in valid_results if r["confidence"] == conf]
                if conf_results:
                    logger.info(f"  {conf}: {len(conf_results)}")

        # 保存结果（非 Full 模式）
        if not args.full:
            results_df = pd.DataFrame(valid_results)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_prefix = "test" if is_test else "valid"
            output_path = OUTPUT_DIR / f"{dataset_prefix}_{timestamp}.csv"
            safe_write_csv(results_df, output_path)
            logger.info(f"结果已保存到: {output_path}")
        else:
            # Full 模式：结果已增量保存
            logger.info(f"结果已增量保存到: {results_path}")

        logger.info("消歧结束")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
