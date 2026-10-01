"""仅特征消融：19 维 LightGBM 不经 LLM 直接做归属，valid 评测

对照问题：NALLM 的能力里，「19 维特征 + 判别器」自己值多少？
三种口径：
  1. anchor  —— luna ③ 仅 LLM（应复现 luna 链路登记的 valid F1 0.9120）
  2. argmax  —— 每条记录对全部候选打分取最大，永远给归属（无 none）
  3. argmax+θ —— 同上，但最高分 < θ_a(0.14) 时判 none（沿用系统已有标定）

用法： uv run python scripts/ablation_features_only.py
"""

import csv
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, "src")
from nallm.utils.evaluate import compute_f1  # noqa: E402

VALID_CSV = Path("archives/valid_luna_llm_only.csv")
VALID_FEATS = Path("results/valid_features_apr.jsonl")
MODEL = Path("models/lgbm_arbiter_19f.txt")
THETA_A = 0.14


def load_gt_and_llm() -> tuple[dict, dict]:
    gt, llm_pred = {}, {}
    with open(VALID_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uid = r["unass_record"]
            gt[uid] = r["ground_truth_author_id"] or "none"
            llm_pred[uid] = r["predicted_author_id"] or "none"
    return gt, llm_pred


def load_features() -> dict[str, list[tuple[str, list[float]]]]:
    feats: dict[str, list[tuple[str, list[float]]]] = {}
    with open(VALID_FEATS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            v = d.get("sim_features", []) + d.get("hand_features", [])
            if "org_available" in d:
                v.append(float(d.get("org_available", 0.0)))
            feats.setdefault(d["unass_record"], []).append((d["candidate_aid"], v))
    return feats


def accuracy(pred: dict, gt: dict) -> float:
    return sum(str(pred[u]).lower() == str(gt[u]).lower() for u in gt) / len(gt)


def none_rate(pred: dict) -> float:
    return sum(v == "none" for v in pred.values()) / len(pred)


def main() -> None:
    gt, llm_pred = load_gt_and_llm()
    feats = load_features()
    booster = lgb.Booster(model_file=str(MODEL))

    truth_none = sum(v == "none" for v in gt.values()) / len(gt)
    print(f"valid 记录 {len(gt)}，特征文件覆盖 {len(feats)}，真值 none 率 {truth_none:.4f}")

    # 批量打分
    rows, owner = [], []
    for rec, cands in feats.items():
        for aid, v in cands:
            rows.append(v)
            owner.append(rec)
    scores = booster.predict(np.array(rows))
    best: dict[str, tuple[float, str]] = {}
    idx = 0
    for rec, cands in feats.items():
        top_s, top_a = -1.0, None
        for aid, _ in cands:
            if scores[idx] > top_s:
                top_s, top_a = scores[idx], aid
            idx += 1
        best[rec] = (float(top_s), top_a)

    variants = {
        "anchor luna-LLM-only": llm_pred,
        "features argmax": {u: best.get(u, (0, "none"))[1] or "none" for u in gt},
        "features argmax+theta": {
            u: (best[u][1] if u in best and best[u][0] >= THETA_A else "none") for u in gt
        },
    }
    print(f"{'口径':24s} {'F1':>7s} {'P':>7s} {'R':>7s} {'Acc':>7s} {'none率':>7s}")
    for name, pred in variants.items():
        f1, p, r, cm = compute_f1(pred, gt)
        print(
            f"{name:24s} {f1:7.4f} {p:7.4f} {r:7.4f} "
            f"{accuracy(pred, gt):7.4f} {none_rate(pred):7.4f}"
        )


if __name__ == "__main__":
    main()
