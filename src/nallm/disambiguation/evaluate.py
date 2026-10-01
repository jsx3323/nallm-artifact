"""消歧评估模块

提供准确率计算等评估功能。
"""


def compute_accuracy(results: list[dict]) -> tuple[int, int, float]:
    """计算准确率

    Args:
        results: 消歧结果列表

    Returns:
        (正确数, 总数, 准确率)
    """
    correct = sum(
        1 for r in results if r["ground_truth_author_id"] == r["predicted_author_id"]
    )
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    return correct, total, accuracy


__all__ = ["compute_accuracy"]
