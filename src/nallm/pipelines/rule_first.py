"""rule_first 链路端到端编排 API。

串联:classify → llm_co → noco → merge,产出 {split}_v4_predictions.csv。
postprocess(inline 训练 + 规则 A/B)本轮不并入(语义与库 PostProcessor 不同)。
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from nallm.classification.subtype_rule_first import classify_rule_first
from nallm.core.config import DATA_DIR, RESULTS_DIR
from nallm.pipelines.llm_co import run_llm_co
from nallm.pipelines.merge import merge_predictions
from nallm.pipelines.noco import run_llm_noco
from nallm.utils.io import load_json


class RuleFirstPaths:
    """rule_first 链路的所有 IO 路径(默认按 split 从 config 推导)。"""

    def __init__(
        self,
        split: str = "valid",
        *,
        filtered_candidates: Path | None = None,
        classified_csv: Path | None = None,
        llm_co_out: Path | None = None,
        noco_out: Path | None = None,
        predictions_out: Path | None = None,
        gt: Path | None = None,
    ) -> None:
        self.split = split
        self.filtered_candidates = filtered_candidates or (
            RESULTS_DIR / f"{split}_filtered_candidates.json"
        )
        self.classified_csv = classified_csv or (
            RESULTS_DIR / f"{split}_unass_classified.csv"
        )
        self.llm_co_out = llm_co_out or (RESULTS_DIR / f"{split}_llm_co_full.json")
        self.noco_out = noco_out or (RESULTS_DIR / f"{split}_noco_co_prompt.json")
        self.predictions_out = predictions_out or (
            RESULTS_DIR / f"{split}_v4_predictions.csv"
        )
        self.gt = gt or (DATA_DIR / split / f"cna_{split}_unass_gt.json")


def step_classify(paths: RuleFirstPaths, *, force: bool = False) -> Path:
    """读 filtered_candidates → classify_rule_first → 写 classified CSV。

    classify 是确定性计算;已有结果且未 force 时跳过。
    """
    if paths.classified_csv.exists() and not force:
        print(f"[skip] {paths.classified_csv} 已存在")
        return paths.classified_csv

    if not paths.filtered_candidates.exists():
        raise FileNotFoundError(f"missing {paths.filtered_candidates}")

    print(f"loading {paths.filtered_candidates}...")
    cands_data = load_json(paths.filtered_candidates)

    sub_counter: Counter = Counter()
    rows = []
    for uid, entry in cands_data.items():
        candidates = entry.get("candidates", [])
        subtype, details = classify_rule_first(candidates)
        sub_counter[subtype] += 1

        top1 = details.get("top1_coauthor")
        strong_count = details.get("strong_count", 0)
        has_competition = 1 if strong_count >= 2 else 0

        rows.append({
            "unass_record": uid,
            "subtype": subtype,
            "num_candidates": len(candidates),
            "position": 0,
            "strength": details.get("reason", ""),
            "has_coauthor": 1 if (top1 and top1 > 0) else 0,
            "has_competition": has_competition,
            "num_competitors": strong_count,
            "title_sim": details.get("top1_title_sim", 0),
            "coauthor_count": top1 or 0,
            "org_verified": 1 if details.get("top1_org_match", 0) > 0 else 0,
        })

    fieldnames = [
        "unass_record", "subtype", "num_candidates", "position", "strength",
        "has_coauthor", "has_competition", "num_competitors",
        "title_sim", "coauthor_count", "org_verified",
    ]
    rows.sort(key=lambda r: r["unass_record"])
    paths.classified_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.classified_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    print(f"\n写入 {paths.classified_csv} ({total} 条)")
    print(f"\n=== {paths.split} rule_first 分类分布 ===")
    for s, n in sub_counter.most_common():
        print(f"  {s}: {n} ({n / total:.1%})")
    auto_n = sub_counter.get("AUTO", 0)
    none_n = sub_counter.get("NONE", 0)
    print(f"\nAUTO+NONE: {auto_n + none_n} ({(auto_n + none_n) / total:.1%})")

    return paths.classified_csv


async def step_llm_co(
    paths: RuleFirstPaths,
    *,
    concurrency: int = 10,
    model: str | None = None,
    force: bool = False,
    n: int = 0,
    seed: int = 42,
    subtypes: list[str] | None = None,
    exclude_gt_none: bool = False,
) -> list[dict]:
    """跑 LLM-Co(resume 语义:run_llm_co 内部跳过已完成)。"""
    return await run_llm_co(
        paths.split,
        out_path=paths.llm_co_out,
        concurrency=concurrency,
        model=model,
        n=n,
        seed=seed,
        subtypes=subtypes,
        exclude_gt_none=exclude_gt_none,
        force_restart=force,
    )


async def step_llm_noco(
    paths: RuleFirstPaths,
    *,
    concurrency: int = 10,
    model: str | None = None,
    force: bool = False,
) -> list[dict]:
    """跑 LLM-NoCo(resume 语义:run_llm_noco 内部跳过已完成)。"""
    return await run_llm_noco(
        paths.split,
        out_path=paths.noco_out,
        concurrency=concurrency,
        model=model,
        force=force,
    )


def step_merge(paths: RuleFirstPaths, *, force: bool = False) -> Path:
    """合并 classify + LLM-Co + LLM-NoCo → predictions CSV(确定性合并)。"""
    return merge_predictions(
        paths.split,
        out_path=paths.predictions_out,
        classified_csv=paths.classified_csv,
        filtered_candidates=paths.filtered_candidates,
        co_path=paths.llm_co_out,
        noco_path=paths.noco_out,
    )


async def run_rule_first(
    split: str = "valid",
    *,
    steps: list[str] | None = None,
    force: bool = False,
    concurrency: int = 10,
    model: str | None = None,
    n: int = 0,
    seed: int = 42,
    subtypes: list[str] | None = None,
    exclude_gt_none: bool = False,
    paths: RuleFirstPaths | None = None,
) -> dict:
    """端到端 rule_first 编排:classify → llm_co → noco → merge。

    Args:
        split: 数据集 (valid/test)
        steps: 要执行的步骤子集(默认全部 ["classify","llm_co","noco","merge"])
        force: 强制重跑(覆盖已有结果)
        concurrency / model: LLM 步骤的并发与模型
        n / seed / subtypes / exclude_gt_none: LLM-Co 抽样/过滤参数
        paths: 自定义路径(默认按 split 推导)

    Returns:
        {step: 结果} 摘要。postprocess 本轮不并入。
    """
    paths = paths or RuleFirstPaths(split=split)
    steps = steps or ["classify", "llm_co", "noco", "merge"]
    summary: dict = {}

    if "classify" in steps:
        summary["classify"] = step_classify(paths, force=force)
    if "llm_co" in steps:
        summary["llm_co"] = await step_llm_co(
            paths,
            concurrency=concurrency,
            model=model,
            force=force,
            n=n,
            seed=seed,
            subtypes=subtypes,
            exclude_gt_none=exclude_gt_none,
        )
    if "noco" in steps:
        summary["noco"] = await step_llm_noco(
            paths, concurrency=concurrency, model=model, force=force
        )
    if "merge" in steps:
        summary["merge"] = step_merge(paths, force=force)

    return summary
