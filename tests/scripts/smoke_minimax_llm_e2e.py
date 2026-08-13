"""冒烟：真实 LLM 端到端多模态（Iter 5 fix）。

目的：构造一次真实 LLM 调用，发送 mm_file://{file_id} URL 给 MiniMax-M3，
验证 provider 真正接受 mm_file:// 引用作为多模态 chat 内容。

跑法：
    PYTHONPATH=src python tests/scripts/smoke_minimax_llm_e2e.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from utils.image_store import ImageStore, MemoryBackend


def main() -> int:
    from config.loader import load_config

    cfg_path = PROJECT_ROOT / "configs" / "local.yaml"
    if not cfg_path.exists():
        cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    cfg = load_config(str(cfg_path))
    api_key = cfg.llm.api_key

    # 1. 构造 ImageStore + 真实尺寸图
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    backend = MemoryBackend(max_memory_items=5, spill_dir=PROJECT_ROOT / "data" / "images")
    store = ImageStore(backend)
    img_url = store.save(img, category="observations")
    print(f"[1] 已存图: {img_url}")

    # 2. 上传到 MiniMax
    mm_url = store.upload_to_minimax(img_url)
    print(f"[2] 上传成功: {mm_url}")

    # 3. 构造 LLM 请求（直接用 OpenAI 协议 image_url 字段，不走 ObserveTool）
    llm = ChatOpenAI(
        base_url=cfg.llm.base_url,
        api_key=api_key,
        model=cfg.llm.model,
        max_tokens=cfg.llm.max_tokens,
        temperature=0,
    )

    messages = [
        SystemMessage(content="你是一个视觉助手。请描述图像内容。"),
        HumanMessage(content=[
            {"type": "text", "text": "这张图里有什么？请简述。"},
            {"type": "image_url", "image_url": {"url": mm_url}},
        ]),
    ]

    print(f"[3] 调 LLM（model={cfg.llm.model}）...")
    try:
        response = llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        print(f"[4] LLM 回复: {content[:500]}")
        if not content.strip():
            print("✗ LLM 回复为空")
            return 1
    except Exception as e:
        print(f"✗ LLM 调用失败: {e}")
        return 1

    print("\n✓ 端到端通过：mm_file:// URL 被 provider 接受并产生有效回复")
    return 0


if __name__ == "__main__":
    sys.exit(main())