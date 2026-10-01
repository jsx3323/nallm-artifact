"""消歧 LLM 配置和 OpenAI SDK 客户端

提供 AsyncOpenAI 客户端、messages 构建和 Token 估算功能。

API 策略:
    - 通用 OpenAI 兼容入口（LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / LLM_EXTRA_BODY），
      默认 DeepSeek 官方端点 + deepseek-v4-flash；OpenRouter 等网关改 env 即切

超时和重试:
    - 全部使用 openai SDK 默认（timeout=600s, max_retries=2 指数退避,
      429 自动遵从 Retry-After），不自定义

用法:
    from nallm.disambiguation.llm import build_messages, chat_completion

    messages = build_messages("p1_high", prompt_data.invoke_params)
    content = await chat_completion(messages)
"""

import json
import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from nallm.disambiguation.prompts import SYSTEM_PROMPT_MAP
from nallm.disambiguation.strategy import DisambiguationStrategy
from deepseek_tokenizer import ds_token

load_dotenv()  # 不 override:shell 已设的变量(如按进程注入的 LLM_API_KEY)优先;override=True 会静默吃掉分片注入

logger = logging.getLogger(__name__)


# ============== LLM 配置 ==============
# 通用 OpenAI 兼容入口（官方 API / OpenRouter / 自建网关均可）:
#   LLM_BASE_URL: 端点，默认 DeepSeek 官方
#   LLM_API_KEY:  密钥（必设）
#   LLM_MODEL:    模型名，默认 deepseek-v4-flash
#                 （OpenRouter 等网关需带厂牌前缀，如 deepseek/deepseek-v4-flash）
#   LLM_EXTRA_BODY: 可选，透传 OpenAI extra_body 扩展参数（JSON），如思考开关
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"


# ============== 纯 OpenAI SDK 客户端 ==============
# human 模板:除已弃用的 p0_high 竞争块模板外,所有线上模板逐字一致(见 prompts.py),
# P0_HIGH 策略实际走 llm_noco 模板,故统一用这一个格式
HUMAN_TEMPLATE = "候选作者列表:\n{candidates}\n\n待消歧论文信息:\n{new_pub}"


class EmptyLLMResponseError(RuntimeError):
    """LLM 返回空 content(思考模型偶发),按错误哨兵处理,防止 parse_output("") 静默当 none 预测"""


_client_instance: AsyncOpenAI | None = None
_extra_body_cache: dict | None = None


def get_client() -> AsyncOpenAI:
    """获取 AsyncOpenAI 单例(LLM_* 环境变量配置的 OpenAI 兼容端点)

    超时/重试全部使用 SDK 默认(timeout=600s, max_retries=2 指数退避,
    429 自动遵从 Retry-After),不自定义。

    Raises:
        ValueError: 缺少 LLM_API_KEY
    """
    global _client_instance
    if _client_instance is None:
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量: LLM_API_KEY")
        # 部分网关强制会话路由头(缺失报 400);OpenAI 官方端点忽略自定义头
        _client_instance = AsyncOpenAI(
            base_url=os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
            api_key=api_key,
            default_headers={"x-opencode-session": os.environ.get("LLM_SESSION", "nallm-v232")},
        )
    return _client_instance


def get_extra_body() -> dict:
    """解析 LLM_EXTRA_BODY(JSON,可选)为透传 OpenAI 的扩展参数

    用法: LLM_EXTRA_BODY='{"reasoning_effort": "max"}'
    未设置返回 {};JSON 非法立即抛 ValueError(启动即暴露,不带病运行)。结果缓存。
    """
    global _extra_body_cache
    cache = _extra_body_cache
    if cache is None:
        raw = os.environ.get("LLM_EXTRA_BODY", "").strip()
        if not raw:
            cache = {}
        else:
            try:
                cache = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM_EXTRA_BODY 不是合法 JSON: {e}") from e
        _extra_body_cache = cache
    return cache


def build_messages(prompt_type: str, invoke_params: dict[str, str]) -> list[dict[str, str]]:
    """构建 chat messages(system + human)

    与原 langchain ChatPromptTemplate 线上行为一致:
    system 取 SYSTEM_PROMPT_MAP(未知类型回退 standard),human 用统一模板,
    invoke_params 中的多余键(如 P0_HIGH 的竞争信息)被忽略,同模板容错行为。
    """
    system = SYSTEM_PROMPT_MAP.get(prompt_type, SYSTEM_PROMPT_MAP["standard"])
    human = HUMAN_TEMPLATE.format(**invoke_params)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": human},
    ]


async def chat_completion(messages: list[dict[str, str]]) -> str:
    """调用 LLM 并返回 content 文本(SDK 默认超时与重试)

    超时/连接错误/429/5xx 全部由 openai SDK 内部按默认策略处理,
    重试耗尽后异常上抛,由 engine 的错误分类统一落 error 哨兵。

    注意: reasoning_effort 必须走 create() 顶层参数——实测部分网关
    (如 zen)对 extra_body 里的 reasoning_effort 不生效,顶层参数正常。

    Raises:
        EmptyLLMResponseError: content 为空
    """
    extra = dict(get_extra_body())  # 拷贝,不改动缓存
    # SDK 一等公民参数提升为顶层,其余键走 extra_body 透传
    top_level = {k: extra.pop(k) for k in ("reasoning_effort",) if k in extra}
    response = await get_client().chat.completions.create(
        model=os.environ.get("LLM_MODEL") or DEFAULT_MODEL,
        messages=messages,
        temperature=0.0,
        extra_body=extra or None,
        **top_level,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise EmptyLLMResponseError("LLM 返回空 content")
    return content


# 策略到 prompt 类型的映射（统一用于 system prompt 选择）
STRATEGY_PROMPT_TYPE_MAP = {
    DisambiguationStrategy.SIMPLIFIED: "simplified",
    DisambiguationStrategy.P0_HIGH: "llm_noco",  # v4 简化版: LLM-NoCo 用新 prompt
    DisambiguationStrategy.P0_MEDIUM: "p0_medium",
    DisambiguationStrategy.P0_LOW: "p0_low",
    DisambiguationStrategy.P2_P3: "p2_p3",
    DisambiguationStrategy.P1_WEAK: "p1_weak",
    DisambiguationStrategy.P1_COMP: "p1_comp",
    DisambiguationStrategy.P1_HIGH: "p1_high",
    DisambiguationStrategy.P1_MEDIUM: "p1_medium",
}

# ============== Token 估算 ==============


def estimate_tokens(text: str) -> int:
    """使用 DeepSeek tokenizer 计算文本的 token 数

    Args:
        text: 待计算的文本

    Returns:
        实际的 token 数
    """
    if not text:
        return 0
    return len(ds_token.encode(text))


def get_system_prompt_by_strategy(strategy: DisambiguationStrategy) -> str:
    """根据策略获取对应的 system prompt

    Args:
        strategy: 消歧策略

    Returns:
        system prompt 文本
    """
    key = STRATEGY_PROMPT_TYPE_MAP.get(strategy, "standard")
    return SYSTEM_PROMPT_MAP.get(key, SYSTEM_PROMPT_MAP["standard"])


def build_human_text(
    strategy: DisambiguationStrategy,
    candidates_text: str,
    new_pub_text: str,
    competition_info: dict | None = None,
) -> str:
    """构建 human message 文本

    Args:
        strategy: 消歧策略
        candidates_text: 候选作者信息文本
        new_pub_text: 待消歧论文信息文本
        competition_info: 竞争检测信息（仅 P0-High 需要）

    Returns:
        格式化的 human message 文本
    """
    if strategy == DisambiguationStrategy.P0_HIGH and competition_info:
        return (
            f"【竞争检测信息】\n"
            f"总候选作者数: {competition_info['num_candidates']}\n"
            f"Top1标题相似度: {competition_info['top1_title_sim']}\n"
            f"Top2标题相似度: {competition_info['top2_title_sim']}\n"
            f"差距: {competition_info['title_sim_gap']}\n"
            f"是否存在竞争: {competition_info['has_competition']}\n\n"
            f"候选作者列表:\n{candidates_text}\n\n"
            f"待消歧论文信息:\n{new_pub_text}"
        )
    return f"候选作者列表:\n{candidates_text}\n\n待消歧论文信息:\n{new_pub_text}"
