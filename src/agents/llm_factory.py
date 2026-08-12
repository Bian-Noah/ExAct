"""LLM 工厂函数：从 LLMConfig 创建 ChatOpenAI 实例。

使用 langchain_openai.ChatOpenAI 通过 OpenAI 兼容接口连接 MiniMax。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from config.loader import LLMConfig


def create_llm(config: LLMConfig) -> ChatOpenAI:
    """从 LLMConfig 创建 ChatOpenAI 实例。

    Args:
        config: LLM 配置（含 api_key/base_url/model/max_tokens）。

    Returns:
        ChatOpenAI 实例，已配置 base_url 和 model。
    """
    return ChatOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=0,
    )
