"""消歧输出模型

定义消歧结果的 Pydantic 模型。
"""

from pydantic import BaseModel, Field


class DisambiguationOutput(BaseModel):
    """消歧输出结构"""

    author_id: str = Field(description="预测的作者ID，无合适作者则为 'none'")
    confidence: str = Field(description="置信度：high/medium/low")
    reasoning: str = Field(description="简要判断理由")


__all__ = ["DisambiguationOutput"]
