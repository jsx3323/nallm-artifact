"""评估工具模块

提供 F1 计算、跨表对比、分层评估等消歧评估功能。

用法:
    from nallm.utils.evaluate import compute_f1, cross_tab, evaluate_stratified

    f1, p, r, cm = compute_f1(uid_to_pred, gt)
    ct = cross_tab(preds_a, preds_b, gt)
    by_type = evaluate_stratified(preds, gt, uid_to_subtype)
"""

from collections import Counter


def compute_f1(uid_to_pred: dict, gt: dict) -> tuple[float, float, float, dict]:
    """计算整体 F1/P/R 和混淆矩阵

    Args:
        uid_to_pred: {uid: predicted_author_id}
        gt: {uid: ground_truth_author_id}（缺失则视为 "none"）

    Returns:
        (F1, P, R, {"TP": int, "FN": int, "FP": int, "TN": int})
    """
    TP = FN = FP = TN = 0
    for uid, pred in uid_to_pred.items():
        g = gt.get(uid, "none")
        if g != "none":
            if str(pred).lower() == str(g).lower():
                TP += 1
            else:
                FN += 1
        else:
            if str(pred).lower() == "none":
                TN += 1
            else:
                FP += 1
    P = TP / (TP + FP) if (TP + FP) else 0
    R = TP / (TP + FN) if (TP + FN) else 0
    F1 = 2 * P * R / (P + R) if (P + R) else 0
    return F1, P, R, {"TP": TP, "FN": FN, "FP": FP, "TN": TN}


def format_metrics(f1: float, p: float, r: float, cm: dict) -> str:
    """格式化指标输出字符串"""
    return (
        f"F1={f1:.4f}  P={p:.4f}  R={r:.4f}  "
        f"TP={cm['TP']} FN={cm['FN']} FP={cm['FP']} TN={cm['TN']}"
    )


def cross_tab(preds_a: dict, preds_b: dict, gt: dict) -> dict:
    """双方法交叉对比

    Args:
        preds_a: 方法 A 的预测结果 {uid: pred_aid}
        preds_b: 方法 B 的预测结果 {uid: pred_bid}
        gt: 真实标签 {uid: gt_aid}

    Returns:
        {
            "common_uids": int,
            "both_right": int,
            "a_better": int,
            "b_better": int,
            "both_wrong": int,
            "a_better_samples": [{uid, gt, a_pred, b_pred}, ...],
            "b_better_samples": [{uid, gt, a_pred, b_pred}, ...],
        }
    """
    common = set(preds_a.keys()) & set(preds_b.keys())
    if not common:
        return {"error": "无共同 UID"}

    a_better = b_better = both_right = both_wrong = 0
    a_better_details = []
    b_better_details = []

    for uid in common:
        pa = preds_a[uid]
        pb = preds_b[uid]
        g = gt.get(uid, "none")
        a_correct = str(pa).lower() == str(g).lower()
        b_correct = str(pb).lower() == str(g).lower()

        if a_correct and b_correct:
            both_right += 1
        elif a_correct and not b_correct:
            a_better += 1
            a_better_details.append({"uid": uid, "gt": g, "a_pred": pa, "b_pred": pb})
        elif not a_correct and b_correct:
            b_better += 1
            b_better_details.append({"uid": uid, "gt": g, "a_pred": pa, "b_pred": pb})
        else:
            both_wrong += 1

    return {
        "common_uids": len(common),
        "both_right": both_right,
        "a_better": a_better,
        "b_better": b_better,
        "both_wrong": both_wrong,
        "a_better_samples": a_better_details[:20],
        "b_better_samples": b_better_details[:20],
    }


def evaluate_stratified(
    uid_to_pred: dict,
    gt: dict,
    uid_to_subtype: dict[str, str],
) -> dict[str, dict]:
    """按子类型分层评估 F1

    Args:
        uid_to_pred: {uid: predicted_author_id}
        gt: {uid: ground_truth_author_id}
        uid_to_subtype: {uid: subtype} 子类型映射

    Returns:
        {subtype: {"f1": float, "p": float, "r": float, "cm": dict, "count": int}}
    """
    by_sub = {}
    for uid, pred in uid_to_pred.items():
        sub = uid_to_subtype.get(uid, "unknown")
        by_sub.setdefault(sub, {})[uid] = pred

    result = {}
    for sub, sub_preds in sorted(by_sub.items()):
        f1, p, r, cm = compute_f1(sub_preds, gt)
        result[sub] = {
            "f1": round(f1, 4),
            "p": round(p, 4),
            "r": round(r, 4),
            "cm": cm,
            "count": len(sub_preds),
        }
    return result


__all__ = [
    "compute_f1",
    "format_metrics",
    "cross_tab",
    "evaluate_stratified",
]
