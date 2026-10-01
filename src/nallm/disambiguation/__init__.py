"""消歧模块 - 策略、Prompt、LLM调用(纯 OpenAI SDK)"""

from nallm.disambiguation.models import DisambiguationOutput
from nallm.disambiguation.strategy import (
    DisambiguationStrategy,
    get_strategy,
    SUBTYPE_STRATEGY_MAP,
)
from nallm.disambiguation.prompts import (
    SYSTEM_PROMPT,
    SIMPLIFIED_SYSTEM_PROMPT,
    SYSTEM_PROMPT_MAP,
)
from nallm.disambiguation.parser import parse_output
from nallm.disambiguation.llm import (
    EmptyLLMResponseError,
    STRATEGY_PROMPT_TYPE_MAP,
    build_messages,
    chat_completion,
    estimate_tokens,
    get_client,
    get_extra_body,
    get_system_prompt_by_strategy,
    build_human_text,
)
from nallm.disambiguation.evaluate import compute_accuracy
from nallm.disambiguation.prompt_builder import (
    PromptData,
    construct_prompt_data,
    create_org_embedding_getter,
    construct_candidate_info,
    construct_unass_pub_info,
    MAX_PUBS_PER_CANDIDATE,
)

__all__ = [
    # Models
    "DisambiguationOutput",
    # Strategy
    "DisambiguationStrategy",
    "get_strategy",
    "SUBTYPE_STRATEGY_MAP",
    # Prompts
    "SYSTEM_PROMPT",
    "SIMPLIFIED_SYSTEM_PROMPT",
    "SYSTEM_PROMPT_MAP",
    # Parser
    "parse_output",
    # LLM (纯 OpenAI SDK)
    "EmptyLLMResponseError",
    "STRATEGY_PROMPT_TYPE_MAP",
    "build_messages",
    "chat_completion",
    "estimate_tokens",
    "get_client",
    "get_extra_body",
    "get_system_prompt_by_strategy",
    "build_human_text",
    # Evaluate
    "compute_accuracy",
    # Prompt Builder
    "PromptData",
    "construct_prompt_data",
    "create_org_embedding_getter",
    "construct_candidate_info",
    "construct_unass_pub_info",
    "MAX_PUBS_PER_CANDIDATE",
]
