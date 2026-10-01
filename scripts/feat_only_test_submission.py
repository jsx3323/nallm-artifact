"""仅特征消融的 test 提交件（规则与 ablation_features_only.py 完全一致）

规则与 scripts/ablation_features_only.py 的 valid 口径完全一致：
19 维特征 + LightGBM v2.3.1，每记录对全部候选打分取 argmax，
最高分 < θ_a(0.14，借用、未调) 判 none；无候选/特征缺行判 none。
不经 LLM、不分层、不后处理——「管线 minus LLM」的极端对照。

输出 RNDeval 提交格式 {"author_id": ["paper_id", ...]}。

用法： uv run python scripts/feat_only_test_submission.py
"""

import csv
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

SRC_CSV = Path("data/test_deepseek_llm_only.csv")  # ③ 快照源，覆盖全部 14,483 条 test 记录
FEATS = Path("data/test_candidate_features.jsonl")  # 19 维特征快照
MODEL = Path("models/lgbm_arbiter_19f.txt")
THETA_A = 0.14
OUT_JSON = Path("archives/test_featonly_argmax_theta.json")
OUT_CSV = Path("archives/test_featonly_argmax_theta.csv")
NONE = "none"


def main() -> None:
    records = [r["unass_record"] for r in csv.DictReader(open(SRC_CSV, encoding="utf-8"))]
    rec_set = set(records)

    feats: dict[str, list[tuple[str, list]]] = {}
    with open(FEATS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["unass_record"] in rec_set:
                v = d.get("sim_features", []) + d.get("hand_features", [])
                if "org_available" in d:
                    v.append(float(d.get("org_available", 0.0)))
                feats.setdefault(d["unass_record"], []).append((d["candidate_aid"], v))

    booster = lgb.Booster(model_file=str(MODEL))
    flat, owner = [], []
    for rec, cands in feats.items():
        for aid, v in cands:
            flat.append(v)
            owner.append(rec)
    scores = booster.predict(np.array(flat))
    best: dict[str, tuple[float, str]] = {}
    idx = 0
    for rec, cands in feats.items():
        top_s, top_a = -1.0, None
        for aid, _ in cands:
            if scores[idx] > top_s:
                top_s, top_a = scores[idx], aid
            idx += 1
        best[rec] = (float(top_s), top_a)

    pred = {}
    for rec in records:
        s, a = best.get(rec, (-1.0, None))
        pred[rec] = a if (a is not None and s >= THETA_A) else NONE

    # 提交 JSON
    submission: dict[str, list[str]] = {}
    for rec, a in pred.items():
        if a == NONE:
            continue
        submission.setdefault(a, []).append(rec.split("-")[0])
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    # 过程 CSV（对照/审计用）
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unass_record", "predicted_author_id", "top1_proba"])
        for rec in records:
            s, a = best.get(rec, (-1.0, None))
            w.writerow([rec, pred[rec], f"{s:.4f}"])

    n_none = sum(v == NONE for v in pred.values())
    n_nofeat = sum(rec not in feats for rec in records)
    print(f"总记录 {len(records)}（特征覆盖 {len(records)-n_nofeat}，缺行 {n_nofeat}）")
    print(f"分配 {len(records)-n_none} / none {n_none}（none 率 {n_none/len(records):.4f}）")
    print(f"提交 JSON: {OUT_JSON}（作者 {len(submission)} 名）")

    # 与 DeepSeek ③ 快照的决策对照（参考）
    snap = {r["unass_record"]: (r["predicted_author_id"] or NONE)
            for r in csv.DictReader(open("archives/test_deepseek_llm_only.csv", encoding="utf-8"))}
    common = [u for u in pred if u in snap]
    agree = sum(pred[u] == snap[u] for u in common)
    print(f"与 ③ 快照决策一致率（参考）: {agree}/{len(common)} = {agree/len(common):.2%}")


if __name__ == "__main__":
    main()
