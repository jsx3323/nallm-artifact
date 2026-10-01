"""v4.1-flash 全量 valid 收尾评估

输入 results/disambiguation/valid_v41_full.csv（13,825 条），产出：
  1. 官方 wF1：仅 LLM / LLM+⑥（θ 0.14/0.20）＋ A/B 触发数
  2. 与 DeepSeek V3.2 快照 / luna 链路的对照（一致率、McNemar）
  3. 归档 archives/valid_v41flash_{llm_only,postprocessed}.csv

用法： uv run python scripts/eval_v41flash_valid.py
"""

import csv
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, "src")

SRC = Path("results/disambiguation/valid_v41_full.csv")
SNAP = Path("archives/valid_deepseek_llm_only.csv")
LUNA = Path("archives/valid_luna_llm_only.csv")
VALID_FEATS = Path("data/valid_candidate_features.jsonl")
MODEL = Path("models/lgbm_arbiter_19f.txt")
THETA_A, THETA_B = 0.14, 0.20
NONE = "none"


def weighted_f1(pred: dict, gt: dict) -> tuple[float, float, float]:
    """官方 weighted-F1（RNDeval 语义，与 replay_postprocess_valid.py 一致）"""
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
    return (2 * wp * wr / (wp + wr) if (wp + wr) > 0 else 0.0), wp, wr


def mcnemar(a: dict, b: dict, gt: dict, keys) -> tuple[int, int, float]:
    x = sum(a[u] == gt[u] and b[u] != gt[u] for u in keys)
    y = sum(a[u] != gt[u] and b[u] == gt[u] for u in keys)
    n = x + y
    p = min(2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2**n, 1.0) if n else 1.0
    return x, y, p


def main() -> None:
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    gt = {r["unass_record"]: r["ground_truth_author_id"] or NONE for r in rows}
    llm = {r["unass_record"]: r["predicted_author_id"] or NONE for r in rows}
    meta = {r["unass_record"]: (r["subtype"], r["strategy"]) for r in rows}

    # 特征 + 模型打分
    feats: dict[str, list[tuple[str, list]]] = defaultdict(list)
    with open(VALID_FEATS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            v = d.get("sim_features", []) + d.get("hand_features", [])
            if "org_available" in d:
                v.append(float(d.get("org_available", 0.0)))
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

    trig_a = trig_b = 0
    gate: dict[str, str] = {}
    for uid, pred in llm.items():
        sub, strat = meta[uid]
        pairs = cand_sorted.get(uid, [])
        if sub == "NONE" or strat == "auto" or pred == "error":
            gate[uid] = pred
            continue
        if pred != NONE and pred in dict(pairs or []) and llm_proba[uid] < THETA_A:
            gate[uid] = NONE
            trig_a += 1
        elif pred == NONE and pairs and pairs[0][1] >= THETA_B:
            gate[uid] = pairs[0][0]
            trig_b += 1
        else:
            gate[uid] = pred

    f1l, pl, rl = weighted_f1(llm, gt)
    f1g, pg, rg = weighted_f1(gate, gt)
    nrl = sum(v == NONE for v in llm.values()) / len(llm)
    print(f"n = {len(rows)}")
    print(f"仅 LLM      wF1 {f1l:.4f}  wP {pl:.4f}  wR {rl:.4f}  none率 {nrl:.4f}")
    print(f"LLM+⑥(θ.14/.20) wF1 {f1g:.4f}  wP {pg:.4f}  wR {rg:.4f}  增益 {f1g-f1l:+.4f}")
    print(f"规则触发 A={trig_a} B={trig_b}（登记: 快照test A=260/B=293, luna valid A=387/B=242→真特征 311/319 量级）")

    # 对照链路
    for name, path in [("V3.2快照", SNAP), ("luna", LUNA)]:
        other = {r["unass_record"]: (r["predicted_author_id"] or NONE) for r in csv.DictReader(open(path, encoding="utf-8"))}
        common = [u for u in gt if u in other]
        agree = sum(llm[u] == other[u] for u in common)
        x, y, p = mcnemar(llm, other, gt, common)
        fo = weighted_f1(other, {u: gt[u] for u in common})[0]
        fn = weighted_f1({u: llm[u] for u in common}, {u: gt[u] for u in common})[0]
        print(f"vs {name}: 交集 {len(common)} 一致率 {agree/len(common):.2%} wF1 {fn:.4f} 对 {fo:.4f} McNemar {x}:{y} p={p:.3f}")

    # 归档
    with open("archives/valid_v41flash_llm_only.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unass_record", "predicted_author_id", "ground_truth_author_id", "subtype", "strategy"])
        for r in rows:
            w.writerow([r["unass_record"], r["predicted_author_id"], r["ground_truth_author_id"], r["subtype"], r["strategy"]])
    with open("archives/valid_v41flash_postprocessed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unass_record", "predicted_author_id", "ground_truth_author_id", "subtype", "strategy"])
        for uid in gt:
            sub, strat = meta[uid]
            w.writerow([uid, gate[uid], gt[uid], sub, strat])
    print("已写 archives/valid_v41flash_{llm_only,postprocessed}.csv")


if __name__ == "__main__":
    main()
