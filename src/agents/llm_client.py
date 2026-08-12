"""MiniMax LLM API 客户端封装。

使用 urllib.request（标准库）实现零依赖 HTTP 调用，
兼容 OpenAI function calling 协议（tools + tool_calls）。
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

from config.loader import LLMConfig


class MiniMaxClient:
    """MiniMax LLM 客户端。

    通过 urllib.request 调用 OpenAI 兼容的 /chat/completions 接口，
    支持文本回复和工具调用（function calling）两种响应模式。

    Attributes:
        config: LLMConfig 实例（含 api_key / model / base_url / max_tokens）。
    """

    def __init__(self, config: LLMConfig):
        """初始化客户端。

        Args:
            config: LLMConfig 实例。不校验 api_key（留待 chat 时校验，
                允许构造时不立即报错，便于测试）。
        """
        self.config = config

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """调用 MiniMax LLM /chat/completions 接口。

        Args:
            messages: OpenAI 消息列表，如
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."}]
            tools: OpenAI function tool schema 列表，如
                [{"type": "function", "function": {...}}]。
                为空或 None 时不传 tools 字段。

        Returns:
            dict，结构：
                {"role": str, "content": str|None,
                 "tool_calls": list[dict], "raw": dict}

        Raises:
            ValueError: api_key 为空或占位符。
            RuntimeError: HTTP 错误 / 网络错误 / 超时 / 响应解析失败。
        """
        # ① 校验 api_key
        api_key = self.config.api_key
        if not api_key or not api_key.strip() or api_key.strip() == "YOUR_API_KEY_HERE":
            raise ValueError(
                "LLM api_key is empty or placeholder, "
                "please fill in configs/local.yaml"
            )

        # ② 组装请求体
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            body["tools"] = tools

        # ③ 组装 urllib Request
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers=headers, method="POST"
        )

        # ④ 发起请求 + 错误捕获
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"LLM HTTP {e.code}: {err_body}") from e
        except urllib.error.URLError as e:
            # socket.timeout 是 URLError 的子类之一，单独优先判断
            if isinstance(e.reason, socket.timeout):
                raise RuntimeError("LLM request timed out (60s)") from e
            raise RuntimeError(f"LLM network error: {e.reason}") from e
        except socket.timeout as e:
            raise RuntimeError("LLM request timed out (60s)") from e
        except TimeoutError as e:
            raise RuntimeError("LLM request timed out (60s)") from e

        # ⑤ 解析响应
        try:
            resp_dict = json.loads(resp_body)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM response JSON parse failed: {e}") from e

        try:
            choice = resp_dict["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"LLM response missing choices[0].message: {resp_dict}"
            ) from e

        # ⑥ 返回标准化结构
        return {
            "role": choice.get("role", "assistant"),
            "content": choice.get("content"),
            "tool_calls": choice.get("tool_calls", []) or [],
            "raw": resp_dict,
        }
