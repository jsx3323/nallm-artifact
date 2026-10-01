"""朴素单提示消融：单一 standard 提示 vs 分层提示的配对比较

同一批 valid 记录，把所有策略的提示强制为 standard（单一立场/单一阈值），
与 v4.1 分层提示结果配对比较。论文口径结果（500 条配对）：朴素 87.20% 对
分层 90.80%，差 +3.60pp，McNemar 6:24，p=0.0014。

用法：
    uv run python scripts/naive_prompt_ablation.py --limit 10    # 试点
    uv run python scripts/naive_prompt_ablation.py --limit 500   # 正式

注:配对基线 archives/valid_v41flash_llm_only.csv 为含 valid 真值的产物,不随仓分发;
本仓运行前需自行生成(先跑 eval_v41flash_valid.py)。
"""

import argparse
import asyncio
import csv
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, "src")

from dotenv import load_dotenv

load_dotenv()  # .env in cwd, if present; credentials may also come from the environment

from nallm.disambiguation.batch import ERROR_SENTINEL  # noqa: E402
from nallm.disambiguation.llm import STRATEGY_PROMPT_TYPE_MAP  # noqa: E402
from nallm.pipelines.llm_first import LlmFirstConfig, RecordFilter, run_llm_first_from_path  # noqa: E402
from nallm.pipelines.progress import make_append_store  # noqa: E402

STRATIFIED = Path("archives/valid_v41flash_llm_only.csv")
NONE = "none"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    # 核心：所有策略 → standard 提示（单一立场/阈值，无分层嘱咐）
    for k in list(STRATEGY_PROMPT_TYPE_MAP):
        STRATEGY_PROMPT_TYPE_MAP[k] = "standard"

    out = Path(args.out or f"results/disambiguation/valid_naive_{args.limit}.csv")
    config = LlmFirstConfig(max_candidates=10, concurrency=8, is_test=False, save_interval=20)
    filt = RecordFilter(limit=args.limit)
    run = await run_llm_first_from_path(
        "valid", results_path=out, config=config, filt=filt,
        on_save=make_append_store(out), on_progress=None,
    )
    results = [r for r in run.results if r.get("predicted_author_id") != ERROR_SENTINEL]
    print(f"处理 {len(results)} 条 -> {out}")

    # 配对比较:同记录 vs 分层提示(v4.1)
    strat = {r["unass_record"]: r for r in csv.DictReader(open(STRATIFIED, encoding="utf-8"))}
    common = [r for r in results if r["unass_record"] in strat]
    def ok_naive(r): return (r["predicted_author_id"] or NONE) == (r["ground_truth_author_id"] or NONE)
    def ok_strat(r):
        s = strat[r["unass_record"]]
        return (s["predicted_author_id"] or NONE) == (s["ground_truth_author_id"] or NONE)
    b = sum(ok_naive(r) and not ok_strat(r) for r in common)
    c = sum(not ok_naive(r) and ok_strat(r) for r in common)
    n = b + c
    p = min(2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n, 1.0) if n else 1.0
    acc_n = sum(ok_naive(r) for r in common) / len(common)
    acc_s = sum(ok_strat(r) for r in common) / len(common)
    agree = sum((r["predicted_author_id"] or NONE) == (strat[r["unass_record"]]["predicted_author_id"] or NONE) for r in common) / len(common)
    print(f"\n配对 n={len(common)}: 朴素 {acc_n:.4f} vs 分层 {acc_s:.4f}  差 {(acc_s-acc_n)*100:+.2f}pp")
    print(f"McNemar {b}:{c} p={p:.4f}  决策一致率 {agree:.2%}")
    band = Counter()
    band_ok = Counter()
    for r in common:
        if r["subtype"] == "NONE" or r["strategy"] == "auto":
            continue
        band[r["subtype"]] += 1
        band_ok[r["subtype"]] += ok_naive(r)
    print("朴素版各子类型(前6):", {k: f"{band_ok[k]}/{v}" for k, v in band.most_common(6)})


if __name__ == "__main__":
    asyncio.run(main())
