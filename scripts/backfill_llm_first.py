"""按目标 ID 集补跑缺失记录(llm_first 补跑入口)。

场景:某次 run 因配额(429)部分失败,只重跑缺失记录并合并成完整结果。
目标界由 --target-csv 的 unass_record 集定义;已完成来源 = --done-csv ∪ --out 自身,
故中断后重跑同一命令即可续传。跑完(非 probe)合并按 target 行序对齐,
便于与对照 run 逐条配对比较。

用法:
    # 探针:先跑 3 条确认配额恢复
    uv run python scripts/backfill_llm_first.py \
        --target-csv results/disambiguation/valid500_<run_a>.csv \
        --done-csv results/disambiguation/valid_20260824_144558.csv \
        --out results/disambiguation/valid500_<run_b>_backfill.csv \
        --final results/disambiguation/valid500_<run_b>_final.csv \
        --probe 3

    # 去掉 --probe 补全量并写 --final
"""

import argparse
import asyncio
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from nallm.disambiguation.batch import ERROR_SENTINEL
from nallm.pipelines.llm_first import LlmFirstConfig, RecordFilter, run_llm_first_from_path
from nallm.pipelines.progress import make_append_store
from nallm.utils.io import safe_write_csv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按目标 ID 集补跑 llm_first 缺失记录")
    parser.add_argument("--target-csv", required=True, type=Path, help="定义目标 ID 界的结果 CSV(行序即合并序)")
    parser.add_argument("--done-csv", type=Path, action="append", default=[], help="已完成记录来源 CSV(可多次传;缺省则目标全集视为缺失)")
    parser.add_argument("--out", required=True, type=Path, help="补跑增量结果文件(续传源之一)")
    parser.add_argument("--final", type=Path, default=None, help="合并输出路径;仅非 probe 且跑满时写")
    parser.add_argument("--probe", type=int, default=None, help="探针模式:只跑前 N 条缺失记录,不合并")
    parser.add_argument("--dataset", "-d", choices=["valid", "test"], default="valid", help="数据集,默认: valid")
    parser.add_argument("--max-candidates", type=int, default=10, help="每条记录的最大候选作者数,默认: 10")
    parser.add_argument("--concurrency", "-c", type=int, default=8, help="并发 worker 数,默认 8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    for noisy in ("httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    target = pd.read_csv(args.target_csv)
    target_order = target["unass_record"].tolist()

    done_ids: set[str] = set()
    for done_path in args.done_csv:
        done_ids |= set(pd.read_csv(done_path, usecols=["unass_record"])["unass_record"])
    if args.out.exists():
        done_ids |= set(pd.read_csv(args.out, usecols=["unass_record"])["unass_record"])

    missing = [r for r in target_order if r not in done_ids]
    logger.info(f"目标 {len(target_order)} 条,已完成 {len(done_ids & set(target_order))} 条,待补 {len(missing)} 条")
    if not missing:
        logger.info("无缺失记录")
    else:
        if args.probe is not None:
            missing = missing[: args.probe]
            logger.info(f"探针模式:只跑前 {len(missing)} 条")

        config = LlmFirstConfig(
            max_candidates=args.max_candidates,
            concurrency=args.concurrency,
            is_test=args.dataset == "test",
            save_interval=20,
        )
        filt = RecordFilter(record_ids=set(missing))
        run = asyncio.run(
            run_llm_first_from_path(
                args.dataset,
                config=config,
                filt=filt,
                on_save=make_append_store(args.out),
            )
        )
        errors = sum(1 for r in run.results if r.get("predicted_author_id") == ERROR_SENTINEL)
        if run.accuracy is not None:
            correct, total, accuracy = run.accuracy
            logger.info(f"本轮: {len(run.results)} 条,错误 {errors},准确率 {correct}/{total} = {accuracy:.2%}")
        else:
            logger.info(f"本轮: {len(run.results)} 条,错误 {errors}")

    # 合并(非 probe;要求目标集已跑满)
    if args.probe is None and args.final is not None:
        out_df = pd.read_csv(args.out) if args.out.exists() else pd.DataFrame()
        out_df = out_df[out_df["predicted_author_id"] != ERROR_SENTINEL]
        merged = {r: row for r, row in zip(out_df["unass_record"], out_df.to_dict("records"))}
        for done_path in args.done_csv:
            df = pd.read_csv(done_path)
            for r, row in zip(df["unass_record"], df.to_dict("records")):
                merged.setdefault(r, row)
        missing_after = [r for r in target_order if r not in merged]
        if missing_after:
            logger.warning(f"仍有 {len(missing_after)} 条缺失,不写 final;可重跑本命令续传")
            return
        final_df = pd.DataFrame([merged[r] for r in target_order])[list(target.columns)]
        safe_write_csv(final_df, args.final)
        logger.info(f"已合并 {len(final_df)} 条 → {args.final}")


if __name__ == "__main__":
    tqdm.monitor_interval = 0  # 无进度条场景避免监控线程
    main()
