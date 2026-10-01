"""姓名索引生成模块

提供姓名索引键生成函数。
"""

from itertools import combinations


def generate_index_keys(name_parts: list[str]) -> set[str]:
    """为姓名生成多个索引 key

    策略：
    1. 考虑姓名顺序的不确定性（中文姓前名后 vs 英文名前姓后）
    2. 生成所有可能的首字母组合（包括子集）
    3. 为每个首字母生成单独的 key（覆盖边缘情况）

    Args:
        name_parts: 标准化后的姓名部分列表

    Returns:
        索引 key 集合
    """
    keys = set()

    if not name_parts:
        return keys

    # 获取所有部分的首字母
    initials = [part[0] for part in name_parts if part]
    if not initials:
        return keys

    # 1. 全首字母组合（原始顺序和反向）
    keys.add("_".join(initials))
    keys.add("_".join(reversed(initials)))

    # 2. 所有部分首字母的排序组合（无序）
    sorted_initials = sorted(initials)
    if len(sorted_initials) >= 2:
        keys.add("_".join(sorted_initials))

    # 3. 生成所有长度 >= 2 的子集组合
    for length in range(2, len(initials) + 1):
        for combo in combinations(range(len(initials)), length):
            subset = [initials[i] for i in combo]
            keys.add("_".join(subset))
            keys.add("_".join(reversed(subset)))
            if len(subset) >= 2:
                keys.add("_".join(sorted(subset)))

    # 4. 每个首字母单独作为 key（覆盖所有可能的单字母匹配）
    for initial in initials:
        keys.add(f"single_{initial}")

    return keys


__all__ = ["generate_index_keys"]
