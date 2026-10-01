"""统计工具模块

提供 TopK 平均等统计功能。
"""


def compute_topk_avg(values: list[float], k: int = 5) -> float:
    """计算 Top-K 平均值

    Args:
        values: 数值列表
        k: 取前 k 个值，默认 5

    Returns:
        Top-K 平均值，如果列表为空则返回 0.0
    """
    if not values:
        return 0.0
    sorted_values = sorted(values, reverse=True)
    top_k = sorted_values[:k]
    return sum(top_k) / len(top_k)


__all__ = ["compute_topk_avg"]
