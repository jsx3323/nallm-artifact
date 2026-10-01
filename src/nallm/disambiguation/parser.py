"""输出解析器

解析 LLM 输出为结构化对象。
"""

import json
import logging
import re

from nallm.disambiguation.models import DisambiguationOutput

logger = logging.getLogger(__name__)


def parse_output(message) -> DisambiguationOutput:
    """解析 LLM 输出为结构化对象

    支持两种格式：
    1. JSON 格式
    2. 键值对格式（DSPy 风格）

    Args:
        message: AIMessage 对象或字符串
    """
    if hasattr(message, "content"):
        text = message.content
    else:
        text = str(message)

    text = text.strip()

    # 尝试解析 JSON
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            # 兼容 LLM-Co prompt 输出的 "answer" 字段
            if "answer" in data and "author_id" not in data:
                data["author_id"] = data.pop("answer")
            if "reason" in data and "reasoning" not in data:
                data["reasoning"] = data.pop("reason")
            # 缺失字段补默认
            data.setdefault("author_id", "none")
            data.setdefault("confidence", "medium")
            data.setdefault("reasoning", "")
            return DisambiguationOutput(**data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.debug("JSON 解析失败，回退到键值对解析: %s", e)

    # 尝试解析键值对格式
    def extract_field(field_name: str, aliases: list[str] | None = None) -> str:
        aliases = aliases or [field_name]
        for alias in aliases:
            pattern = rf"(?:^|\n)\s*{alias}\s*[:：]\s*(.+?)(?=\n\s*\w+\s*[:：]|\Z)"
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    author_id = extract_field("author_id", ["author_id", "作者ID", "预测作者", "answer"])
    confidence = extract_field("confidence", ["confidence", "置信度"])
    reasoning = extract_field("reasoning", ["reasoning", "理由", "判断理由", "reason"])

    if not author_id:
        id_match = re.search(r"\b[a-zA-Z0-9]{8}\b", text)
        author_id = id_match.group() if id_match else "none"

    if not confidence:
        if "high" in text.lower():
            confidence = "high"
        elif "low" in text.lower():
            confidence = "low"
        else:
            confidence = "medium"

    return DisambiguationOutput(
        author_id=author_id,
        confidence=confidence,
        reasoning=reasoning,
    )


__all__ = ["parse_output"]
