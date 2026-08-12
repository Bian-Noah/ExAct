"""MiniMaxClient 单元测试。

覆盖：构造校验时机、正常文本/工具调用响应、HTTP/网络/超时错误、响应解析失败。
全部使用 unittest.mock.patch 拦截 urllib.request.urlopen，不发真实请求。
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from agents.llm_client import MiniMaxClient
from config.loader import LLMConfig


def _make_response(body: dict | str) -> MagicMock:
    """构造一个可作 urlopen context manager 的 mock，read() 返回 body bytes。"""
    if isinstance(body, str):
        payload = body.encode("utf-8")
    else:
        payload = json.dumps(body).encode("utf-8")
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.read = MagicMock(return_value=payload)
    return mock


# ---------- 1. 构造与校验时机 ----------

def test_construct_with_empty_api_key_does_not_raise():
    """构造时 api_key 为空不抛异常（留待 chat 时校验）。"""
    client = MiniMaxClient(LLMConfig(api_key=""))
    assert client.config.api_key == ""


def test_chat_raises_value_error_when_api_key_empty():
    client = MiniMaxClient(LLMConfig(api_key=""))
    with pytest.raises(ValueError, match="api_key is empty"):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_value_error_when_api_key_is_placeholder():
    client = MiniMaxClient(LLMConfig(api_key="YOUR_API_KEY_HERE"))
    with pytest.raises(ValueError, match="api_key is empty"):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_value_error_when_api_key_only_whitespace():
    client = MiniMaxClient(LLMConfig(api_key="   "))
    with pytest.raises(ValueError, match="api_key is empty"):
        client.chat([{"role": "user", "content": "hi"}])


# ---------- 2. 正常文本响应 ----------

def test_normal_text_response():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    resp_body = {
        "choices": [
            {"message": {"role": "assistant", "content": "任务完成"}}
        ]
    }
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)) as m:
        result = client.chat([{"role": "user", "content": "做任务"}])

    assert result["role"] == "assistant"
    assert result["content"] == "任务完成"
    assert result["tool_calls"] == []
    assert result["raw"] == resp_body

    # 校验 Request 参数
    req = m.call_args[0][0]
    assert req.get_method() == "POST"
    assert req.full_url == "https://api.minimax.chat/v1/chat/completions"
    assert req.headers.get("Content-type") == "application/json"
    assert req.headers.get("Authorization") == "Bearer sk-test"
    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "MiniMax-M3"
    assert body["messages"] == [{"role": "user", "content": "做任务"}]
    assert body["max_tokens"] == 2048


def test_tools_not_sent_when_empty():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    resp_body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)) as m:
        client.chat([{"role": "user", "content": "hi"}], tools=[])
    body = json.loads(m.call_args[0][0].data.decode("utf-8"))
    assert "tools" not in body


def test_tools_sent_when_provided():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    resp_body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    tools_schema = [{"type": "function", "function": {"name": "observe"}}]
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)) as m:
        client.chat([{"role": "user", "content": "hi"}], tools=tools_schema)
    body = json.loads(m.call_args[0][0].data.decode("utf-8"))
    assert body["tools"] == tools_schema


# ---------- 3. 正常工具调用响应 ----------

def test_normal_tool_call_response():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    resp_body = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "observe", "arguments": "{\"target\": \"red_block\"}"}
                }]
            }
        }]
    }
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)):
        result = client.chat([{"role": "user", "content": "看场景"}])

    assert result["content"] is None
    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["function"]["name"] == "observe"
    assert tc["function"]["arguments"] == "{\"target\": \"red_block\"}"


# ---------- 4. HTTP 错误 ----------

def test_http_error_401():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    err = urllib.error.HTTPError(
        url="https://api.minimax.chat/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=io.BytesIO(b'{"error": "invalid api key"}'),
    )
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="HTTP 401"):
            client.chat([{"role": "user", "content": "hi"}])


def test_http_error_500_with_body():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    err = urllib.error.HTTPError(
        url="",
        code=500,
        msg="Internal Server Error",
        hdrs=None,
        fp=io.BytesIO(b'{"error": "server boom"}'),
    )
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="HTTP 500.*server boom"):
            client.chat([{"role": "user", "content": "hi"}])


# ---------- 5. 网络错误 ----------

def test_url_error_network():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    err = urllib.error.URLError(reason="Connection refused")
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="network error"):
            client.chat([{"role": "user", "content": "hi"}])


# ---------- 6. 超时 ----------

def test_socket_timeout_via_url_error():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    # urllib 把 socket.timeout 包装在 URLError.reason 中
    err = urllib.error.URLError(reason=socket.timeout("timed out"))
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(RuntimeError, match="timed out"):
            client.chat([{"role": "user", "content": "hi"}])


def test_socket_timeout_direct():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
        with pytest.raises(RuntimeError, match="timed out"):
            client.chat([{"role": "user", "content": "hi"}])


# ---------- 7. 响应解析失败 ----------

def test_response_missing_choices():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    resp_body = {"error": "something went wrong"}
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)):
        with pytest.raises(RuntimeError, match="missing choices"):
            client.chat([{"role": "user", "content": "hi"}])


def test_response_choices_empty_list():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    resp_body = {"choices": []}
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)):
        with pytest.raises(RuntimeError, match="missing choices"):
            client.chat([{"role": "user", "content": "hi"}])


def test_response_not_json():
    client = MiniMaxClient(LLMConfig(api_key="sk-test"))
    with patch("urllib.request.urlopen", return_value=_make_response("not json <<<")):
        with pytest.raises(RuntimeError, match="JSON parse failed"):
            client.chat([{"role": "user", "content": "hi"}])


# ---------- 8. 自定义 base_url 与 model ----------

def test_custom_base_url_and_model():
    cfg = LLMConfig(api_key="sk-x", model="CustomModel", base_url="https://example.com/api")
    client = MiniMaxClient(cfg)
    resp_body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    with patch("urllib.request.urlopen", return_value=_make_response(resp_body)) as m:
        client.chat([{"role": "user", "content": "hi"}])
    req = m.call_args[0][0]
    assert req.full_url == "https://example.com/api/chat/completions"
    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "CustomModel"
