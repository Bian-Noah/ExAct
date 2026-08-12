"""llm_factory 单元测试。"""

import pytest
from langchain_openai import ChatOpenAI

from agents.llm_factory import create_llm
from config.loader import LLMConfig


class TestCreateLlm:
    """create_llm 工厂函数测试。"""

    def test_returns_chat_openai_instance(self):
        """create_llm 返回 ChatOpenAI 实例。"""
        config = LLMConfig(
            api_key="test-key",
            model="MiniMax-M3",
            base_url="https://api.minimax.chat/v1",
            max_tokens=2048,
        )
        llm = create_llm(config)
        assert isinstance(llm, ChatOpenAI)

    def test_model_matches_config(self):
        """model 匹配 config。"""
        config = LLMConfig(
            api_key="test-key",
            model="MiniMax-M3",
            base_url="https://api.minimax.chat/v1",
            max_tokens=1024,
        )
        llm = create_llm(config)
        assert llm.model_name == "MiniMax-M3"

    def test_base_url_matches_config(self):
        """base_url 匹配 config。"""
        config = LLMConfig(
            api_key="test-key",
            model="MiniMax-M3",
            base_url="https://api.minimax.chat/v1",
            max_tokens=2048,
        )
        llm = create_llm(config)
        assert llm.openai_api_base == "https://api.minimax.chat/v1"

    def test_placeholder_key_does_not_raise(self):
        """占位符 key 不抛异常（ChatOpenAI 构造时不校验）。"""
        config = LLMConfig(
            api_key="YOUR_API_KEY_HERE",
            model="MiniMax-M3",
            base_url="https://api.minimax.chat/v1",
            max_tokens=2048,
        )
        llm = create_llm(config)
        assert isinstance(llm, ChatOpenAI)
