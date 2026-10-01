"""rule_first 链路合并:把 classify / LLM-Co / LLM-NoCo 结果路由成 predictions CSV。

下沉自 scripts/merge_llm_results.py。source 路由:
- AUTO : 规则判 top1
- NONE : 规则判 none
- LLM-Co / LLM-NoCo : LLM 判
"""

from __future__ import annotations

import csv
from pathlib import Path

from nallm.core.config import RESULTS_DIR
from nallm.utils.io import load_json


def top1_aid(candidates: list[dict]) -> str:
    """规则选 top1 候选作者(合作者>机构>会议>标题排序)。"""
    if not candidates:
        return "none"

    def _aid(c):
        return c.get("aid") or c.get("candidate_aid")

    def _org(c):
        return c.get("org_sim", c.get("org_sim_max", c.get("main_org_sim", 0)))

    def _venue(c):
        return c.get("venue_sim", c.get("venue_sim_max", 0))

    s = sorted(
        candidates,
        key=lambda c: (
            -(c.get("coauthor_count", 0) > 0),
            -(c.get("coauthor_org_match_count", 0) > 0),
            -_org(c),
            -_venue(c),
            -c.get("title_sim_max", c.get("title_sim", 0)),
        ),
    )
    return _aid(s[0]) or "none"


def merge_predictions(
    split: str = "valid",
    *,
    out_path: Path | None = None,
    classified_csv: Path | None = None,
    filtered_candidates: Path | None = None,
    co_path: Path | None = None,
    noco_path: Path | None = None,
) -> Path:
    """合并 classify + LLM-Co + LLM-NoCo → predictions CSV。

    输出列:uid, predicted_aid, subtype, source。
    LLM-Co 数据源优先 {split}_llm_co_full.json,其次 _llm_all_reparsed.json 兜底。

    Returns:
        输出 CSV 路径。
    """
    out_path = out_path or (RESULTS_DIR / f"{split}_v4_predictions.csv")
    classified_csv = classified_csv or (RESULTS_DIR / f"{split}_unass_classified.csv")
    filtered_candidates = (
        filtered_candidates or RESULTS_DIR / f"{split}_filtered_candidates.json"
    )
    co_path = co_path or (RESULTS_DIR / f"{split}_llm_co_full.json")
    co_path_reparsed = RESULTS_DIR / f"{split}_llm_all_reparsed.json"
    co_path_used = co_path if co_path.exists() else co_path_reparsed
    noco_path = noco_path or (RESULTS_DIR / f"{split}_noco_co_prompt.json")

    # 加载分类
    uid2sub: dict[str, str] = {}
    with open(classified_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid2sub[row["unass_record"]] = row["subtype"]

    # 加载 candidates
    cands_data = load_json(filtered_candidates)

    # 加载 LLM-Co 结果（优先 _llm_co_full.json，其次 reparsed 兜底）
    co_results: dict[str, dict] = {}
    if co_path_used.exists():
        for r in load_json(co_path_used):
            co_results[r["case_id"]] = r
    print(f"LLM 数据源: {co_path_used.name} ({len(co_results)} 条)")

    # 加载 LLM-NoCo 结果
    noco_results: dict[str, dict] = {}
    if noco_path.exists():
        for r in load_json(noco_path):
            noco_results[r["case_id"]] = r

    # 合并
    rows = []
    source_counter = {"AUTO": 0, "NONE": 0, "LLM-Co": 0, "LLM-NoCo": 0}
    for uid, sub in uid2sub.items():
        if sub == "AUTO":
            pred = top1_aid(cands_data.get(uid, {}).get("candidates", []))
            source = "AUTO"
        elif sub == "NONE":
            pred = "none"
            source = "NONE"
        elif sub == "LLM-Co":
            r = co_results.get(uid)
            pred = r["pred"] if r else "none"
            source = "LLM-Co"
        elif sub == "LLM-NoCo":
            r = noco_results.get(uid)
            pred = r["pred"] if r else "none"
            source = "LLM-NoCo"
        else:
            continue
        source_counter[source] += 1
        rows.append({"uid": uid, "predicted_aid": pred, "subtype": sub, "source": source})

    rows.sort(key=lambda r: r["uid"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "predicted_aid", "subtype", "source"])
        w.writeheader()
        w.writerows(rows)

    print(f"输出: {out_path} ({len(rows)} 条)")
    for k, n in source_counter.items():
        print(f"  {k}: {n} ({n / len(rows):.1%})" if rows else f"  {k}: {n}")

    return out_path
