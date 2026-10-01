"""valid ⑥ 后处理的无泄露评估：对半 θ 迁移

θ_a/θ_b 是 valid 网格出来的（泄露源）；LightGBM 训于 train、特征推理时算，均无泄露。
本脚本把 valid 按 crc32 对半切：A 半网格选 θ* → B 半评估，B 半选 → A 半评估，
选参与评估不见面。对照冻结 θ(0.14/0.20) 的同半表现，差值即 θ 选择过拟合幅度。

用法： uv run python scripts/theta_transfer_valid.py [llm_csv]
默认 llm_csv = archives/valid_v41flash_llm_only.csv
"""

import csv
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, "src")

LLM_CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "archives/valid_v41flash_llm_only.csv")
VALID_FEATS = Path("data/valid_candidate_features.jsonl")
MODEL = Path("models/lgbm_arbiter_19f.txt")
NONE = "none"


def weighted_f1(pred: dict, gt: dict) -> float:
    gt_sets, pred_sets = defaultdict(set), defaultdict(set)
    for uid, a in gt.items():
        if a != NONE:
            gt_sets[a].add(uid)
    for uid, a in pred.items():
        if a != NONE:
            pred_sets[a].add(uid)
    total = sum(len(v) for v in gt_sets.values())
    wp = wr = 0.0
    for a, gp in gt_sets.items():
        pp = pred_sets.get(a, set())
        inter = len(gp & pp)
        wp += (inter / max(len(pp), 1)) * len(gp) / total
        wr += (inter / max(len(gp), 1)) * len(gp) / total
    return 2 * wp * wr / (wp + wr) if (wp + wr) > 0 else 0.0


def main() -> None:
    rows = list(csv.DictReader(open(LLM_CSV, encoding="utf-8")))
    gt = {r["unass_record"]: r["ground_truth_author_id"] or NONE for r in rows}
    llm = {r["unass_record"]: r["predicted_author_id"] or NONE for r in rows}
    meta = {r["unass_record"]: (r["subtype"], r["strategy"]) for r in rows}

    feats: dict[str, list[tuple[str, list]]] = defaultdict(list)
    with open(VALID_FEATS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            v = d.get("sim_features", []) + d.get("hand_features", []) + (
                [float(d.get("org_available", 0.0))] if "org_available" in d else [])
            feats[d["unass_record"]].append((d["candidate_aid"], v))
    booster = lgb.Booster(model_file=str(MODEL))
    flat, aid_flat, ptr = [], [], {}
    for rec, cands in feats.items():
        ptr[rec] = (len(flat), len(flat) + len(cands))
        for aid, v in cands:
            flat.append(v)
            aid_flat.append(aid)
    scores = booster.predict(np.array(flat))
    cand_sorted, llm_proba = {}, {}
    for rec, (s, e) in ptr.items():
        pairs = sorted(zip(aid_flat[s:e], scores[s:e].tolist()), key=lambda t: -t[1])
        cand_sorted[rec] = pairs
        llm_proba[rec] = dict(pairs).get(llm.get(rec, NONE), 0.0)

    def replay(theta_a: float, theta_b: float, keys) -> dict:
        out = {}
        for uid in keys:
            sub, strat = meta[uid]
            pred = llm[uid]
            if sub == "NONE" or strat == "auto" or pred == "error":
                out[uid] = pred
                continue
            pairs = cand_sorted.get(uid, [])
            if pred != NONE and pred in dict(pairs or []) and llm_proba[uid] < theta_a:
                out[uid] = NONE
            elif pred == NONE and pairs and pairs[0][1] >= theta_b:
                out[uid] = pairs[0][0]
            else:
                out[uid] = pred
        return out

    halves = {"H0": [], "H1": []}
    for uid in gt:
        halves["H0" if zlib.crc32(uid.encode()) % 2 == 0 else "H1"].append(uid)

    # 各半网格选 θ*（0.00–0.30 步 0.01）
    grid = [(round(a, 2), round(b, 2)) for a in np.arange(0, 0.301, 0.01) for b in np.arange(0, 0.301, 0.01)]
    theta_star, f1_base = {}, {}
    for h, keys in halves.items():
        best, best_f1 = None, -1.0
        gth = {u: gt[u] for u in keys}
        for ta, tb in grid:
            f1 = weighted_f1(replay(ta, tb, keys), gth)
            if f1 > best_f1:
                best, best_f1 = (ta, tb), f1
        theta_star[h], f1_base[h] = best, weighted_f1({u: llm[u] for u in keys}, gth)
        print(f"{h}: n={len(keys)} 仅③ {f1_base[h]:.4f} | 网格内最优 θ*={best} {best_f1:.4f}")

    print()
    frozen = (0.14, 0.20)
    for tune_h, eval_h in (("H0", "H1"), ("H1", "H0")):
        keys = halves[eval_h]
        gth = {u: gt[u] for u in keys}
        f1_tr = weighted_f1(replay(*theta_star[tune_h], keys), gth)
        f1_fr = weighted_f1(replay(*frozen, keys), gth)
        print(f"θ*({tune_h})={theta_star[tune_h]} → {eval_h} 评估: {f1_tr:.4f}（+{f1_tr-f1_base[eval_h]:.4f}） | "
              f"冻结(0.14,0.20): {f1_fr:.4f}（+{f1_fr-f1_base[eval_h]:.4f}） | 迁移落差 {f1_fr-f1_tr:+.4f}")

    tr = np.mean([weighted_f1(replay(*theta_star[t], halves[e]), {u: gt[u] for u in halves[e]})
                  for t, e in (("H0", "H1"), ("H1", "H0"))])
    fr = np.mean([weighted_f1(replay(*frozen, halves[e]), {u: gt[u] for u in halves[e]}) for e in ("H0", "H1")])
    bs = np.mean([f1_base[e] for e in ("H0", "H1")])
    print(f"\n两向平均: 仅③ {bs:.4f} → 迁移θ {tr:.4f}（+{tr-bs:.4f}） / 冻结θ {fr:.4f}（+{fr-bs:.4f}）")
    print(f"θ 选择过拟合幅度（冻结 − 迁移）: {fr-tr:+.4f}")


if __name__ == "__main__":
    main()
