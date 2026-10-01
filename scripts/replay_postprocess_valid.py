"""valid 侧 ⑥ 后处理重放与消融（单脚本多口径）

产出：
  1. 官方 weighted-F1 口径下四对比：仅 LLM / LLM+⑥ / 仅特征 argmax / argmax+θ
  2. θ_a × θ_b 敏感性网格 → outputs/theta_grid_valid.csv（图用）
  3. 分层准确率（14 个 LLM 路由子类型，仅 LLM 对 +⑥）→ outputs/stratified_valid.csv（图 3 用）
  4. 重放生成的 valid_luna_postprocessed → archives/（补归档缺口）

校验锚点：仅 LLM 应 ≈ 0.9120，LLM+⑥ 应 ≈ 0.9412（官方口径 valid F1 登记值）。

用法： uv run python scripts/replay_postprocess_valid.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, "src")

VALID_CSV = Path("archives/valid_luna_llm_only.csv")
VALID_FEATS = Path("data/valid_candidate_features.jsonl")  # org_available 取真值逻辑
MODEL = Path("models/lgbm_arbiter_19f.txt")
THETA_A, THETA_B = 0.14, 0.20
NONE = "none"


def weighted_f1(pred: dict, gt: dict) -> tuple[float, float, float]:
    """官方 weighted-F1（THUDM WhoIsWho RNDeval 语义，按真值作者支撑加权宏 P/R）"""
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
        p = inter / max(len(pp), 1)
        r = inter / max(len(gp), 1)
        w = len(gp) / total
        wp += p * w
        wr += r * w
    f1 = 2 * wp * wr / (wp + wr) if (wp + wr) > 0 else 0.0
    return f1, wp, wr


def accuracy(pred: dict, gt: dict) -> float:
    return sum(pred[u] == gt[u] for u in gt) / len(gt)


def main() -> None:
    rows = list(csv.DictReader(open(VALID_CSV, encoding="utf-8")))
    gt = {r["unass_record"]: r["ground_truth_author_id"] or NONE for r in rows}
    llm = {r["unass_record"]: r["predicted_author_id"] or NONE for r in rows}
    meta = {r["unass_record"]: (r["subtype"], r["strategy"]) for r in rows}

    feats: dict[str, list[tuple[str, list]]] = defaultdict(list)
    with open(VALID_FEATS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            v = d.get("sim_features", []) + d.get("hand_features", [])
            if "org_available" in d:
                v.append(float(d.get("org_available", 0.0)))
            feats[d["unass_record"]].append((d["candidate_aid"], v))

    booster = lgb.Booster(model_file=str(MODEL))
    flat, owner, aid_flat, ptr = [], [], [], {}
    for rec, cands in feats.items():
        ptr[rec] = (len(flat), len(flat) + len(cands))
        for aid, v in cands:
            flat.append(v)
            aid_flat.append(aid)
    scores = booster.predict(np.array(flat))

    # 每记录预计算：降序候选、LLM 所选候选分数、top1
    cand_sorted, llm_proba, top1 = {}, {}, {}
    for rec, (s, e) in ptr.items():
        pairs = sorted(zip(aid_flat[s:e], scores[s:e].tolist()), key=lambda x: -x[1])
        cand_sorted[rec] = pairs
        top1[rec] = pairs[0] if pairs else (NONE, 0.0)
        pmap = dict(pairs)
        llm_proba[rec] = pmap.get(llm.get(rec, NONE), 0.0)

    def replay(theta_a: float, theta_b: float) -> dict:
        out = {}
        for uid, pred in llm.items():
            sub, strat = meta[uid]
            if sub == "NONE" or strat == "auto" or pred == "error":
                out[uid] = pred
                continue
            pairs = cand_sorted.get(uid, [])
            if pred != NONE and pred in dict(pairs or []) and llm_proba[uid] < theta_a:
                out[uid] = NONE  # 规则 A
            elif pred == NONE and pairs and pairs[0][1] >= theta_b:
                out[uid] = pairs[0][0]  # 规则 B
            else:
                out[uid] = pred
        return out

    feat_argmax = {u: top1.get(u, (NONE, 0.0))[0] for u in gt}
    feat_theta = {
        u: (top1[u][0] if u in top1 and top1[u][1] >= THETA_A else NONE) for u in gt
    }

    variants = {
        "llm_only": llm,
        "llm_gate(0.14/0.20)": replay(THETA_A, THETA_B),
        "feat_argmax": feat_argmax,
        f"feat_argmax+theta({THETA_A})": feat_theta,
    }
    print(f"{'口径':22s} {'wF1':>7s} {'wP':>7s} {'wR':>7s} {'Acc':>7s} {'none率':>7s}")
    for name, pred in variants.items():
        f1, p, r = weighted_f1(pred, gt)
        nr = sum(v == NONE for v in pred.values()) / len(pred)
        print(f"{name:22s} {f1:7.4f} {p:7.4f} {r:7.4f} {accuracy(pred, gt):7.4f} {nr:7.4f}")

    # θ 网格
    grid = []
    for ta in np.arange(0.0, 0.301, 0.01):
        for tb in np.arange(0.0, 0.301, 0.01):
            pred = replay(round(ta, 2), round(tb, 2))
            f1, _, _ = weighted_f1(pred, gt)
            grid.append((round(ta, 2), round(tb, 2), round(f1, 6)))
    with open("outputs/theta_grid_valid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta_a", "theta_b", "weighted_f1"])
        w.writerows(grid)
    best = max(grid, key=lambda x: x[2])
    cur = [g for g in grid if abs(g[0] - THETA_A) < 1e-9 and abs(g[1] - THETA_B) < 1e-9][0]
    print(f"θ 网格: 当前 (0.14,0.20) F1={cur[2]:.4f}；网格最优 {best}（泄露口径，仅看形状）")

    # 分层准确率（仅 LLM 对 + gate）
    gate = variants["llm_gate(0.14/0.20)"]
    strat_acc = defaultdict(lambda: [0, 0, 0])  # n, llm对, gate对
    for uid in gt:
        sub, strat = meta[uid]
        if sub == "NONE" or strat == "auto":
            continue
        s = strat_acc[sub]
        s[0] += 1
        s[1] += llm[uid] == gt[uid]
        s[2] += gate[uid] == gt[uid]
    with open("outputs/stratified_valid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subtype", "n", "acc_llm_only", "acc_gated"])
        for sub, (n, a, b) in sorted(strat_acc.items(), key=lambda x: -x[1][0]):
            w.writerow([sub, n, round(a / n, 4), round(b / n, 4)])
    print("分层子类型数:", len(strat_acc), "（LLM 路由口径，应 14）")

    # 补归档：重放生成的 valid 后处理件
    with open("archives/valid_luna_postprocessed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unass_record", "predicted_author_id", "ground_truth_author_id", "subtype", "strategy"])
        for uid in gt:
            sub, strat = meta[uid]
            w.writerow([uid, gate[uid], gt[uid], sub, strat])
    print("已写 archives/valid_luna_postprocessed.csv")


if __name__ == "__main__":
    main()
