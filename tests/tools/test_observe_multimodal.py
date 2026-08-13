"""ObserveTool 多模态 content blocks 单元测试（Iteration 5）。

覆盖（按 tasks.md 1.2.1~1.2.8）：
- 1.2.1 test_observe_tool_text_only_when_image_store_none
- 1.2.2 test_observe_tool_image_block_when_image_store_present
- 1.2.3 test_observe_tool_degrades_without_rgb
- 1.2.4 test_observe_tool_degrades_without_image_store
- 1.2.5 test_observe_tool_recorder_emit_still_works_with_image_store
- 1.2.6 test_observe_tool_mock_backend_records_save_params
- 1.2.7 test_observe_tool_target_filter_with_image_store
- 1.2.8 test_observe_tool_text_block_preserves_formatting

核心约定（Iteration 5）：
- ObserveTool._run() 返回 list[dict]（LangChain 标准 content blocks）
- 首个块永远是 {"type": "text", "text": "..."}
- 第二块（条件）是 {"type": "image", "url": "img://..."}，仅当
  image_store 已注入且 obs['rgb'] 可用时出现
- image 块字段是扁平的（无嵌套），url 字段直接放 URL 字符串
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from tools.observe import ObserveInput, ObserveTool, _format_text_lines
from utils.image_store import ImageStore, MemoryBackend, StorageBackend
from utils.image_store.url_scheme import parse_url


# ============================================================================
# FakeEnv（duck typing env）
# ============================================================================


class FakeEnv:
    """duck typing 环境，可配置 get_obs 返回值与是否含 rgb 字段。"""

    def __init__(self, obs: dict, include_rgb: bool = True):
        self._obs = obs
        self._include_rgb = include_rgb

    def get_obs(self, include_rgb: bool = True) -> dict:
        if not self._include_rgb:
            # 移除 rgb 字段
            return {k: v for k, v in self._obs.items() if k != "rgb"}
        return self._obs


def _default_obs(rgb: np.ndarray | None = None) -> dict:
    """构造默认 obs dict（含 ee_pos + object_info + 可选 rgb）。"""
    obs = {
        "object_info": [
            {"name": "red_block", "position": (0.1, 0.2, 0.05)},
            {"name": "blue_box", "position": (0.3, 0, 0.03)},
        ],
        "ee_pos": (0.0, 0.0, 0.5),
    }
    if rgb is not None:
        obs["rgb"] = rgb
    return obs


# ============================================================================
# 1.2.1: image_store=None 时只返回 text 块
# ============================================================================


def test_observe_tool_text_only_when_image_store_none():
    """image_store=None 时返回 list[dict] 长度 1，仅含 text 块。"""
    env = FakeEnv(_default_obs())
    tool = ObserveTool(env=env)  # image_store 默认为 None

    result = tool._run()

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert "末端执行器位置" in result[0]["text"]
    assert "场景物体列表" in result[0]["text"]
    # 无 image 块
    assert not any(b.get("type") == "image" for b in result)


# ============================================================================
# 1.2.2: 注入 ImageStore 时追加 image 块
# ============================================================================


def test_observe_tool_image_block_when_image_store_present(tmp_path: Path):
    """注入 ImageStore + rgb 可用 → 返回 list 长度 2，第二块是 image 块。

    upload_to_minimax 被 mock 掉（返回 mock 字符串），避免单测真打 MiniMax。
    """
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    env = FakeEnv(_default_obs(rgb=rgb))
    image_store = ImageStore(
        MemoryBackend(max_memory_items=5, spill_dir=tmp_path)
    )
    # Mock upload_to_minimax 避免真打 MiniMax（真实端到端在 smoke 脚本里）
    image_store.upload_to_minimax = MagicMock(return_value="mm_file://test_fake_id")
    tool = ObserveTool(env=env, image_store=image_store)

    result = tool._run()

    # 返回 list 长度 2
    assert isinstance(result, list)
    assert len(result) == 2

    # 第一个块：text
    assert result[0]["type"] == "text"
    assert "末端执行器位置" in result[0]["text"]

    # 第二个块：image，url 来自 mock 的 upload_to_minimax
    assert result[1]["type"] == "image"
    assert "url" in result[1]
    image_url = result[1]["url"]
    assert image_url == "mm_file://test_fake_id"  # mock 返回值

    # upload_to_minimax 被调 1 次，参数是 save 返回的 img:// URL
    image_store.upload_to_minimax.assert_called_once()
    called_url = image_store.upload_to_minimax.call_args[0][0]
    assert called_url.startswith("img://observations/")

    # 验证 image_store 仍可加载原图
    assert image_store.exists(called_url) is True
    loaded = image_store.load(called_url)
    assert loaded.shape == (480, 640, 3)


# ============================================================================
# 1.2.3: rgb 缺失时降级
# ============================================================================


def test_observe_tool_degrades_without_rgb(tmp_path: Path):
    """FakeEnv.get_obs() 返回无 'rgb' 字段的 dict → list 长度 1，save 未被调用。"""
    env = FakeEnv(_default_obs(), include_rgb=False)
    image_store = ImageStore(
        MemoryBackend(max_memory_items=5, spill_dir=tmp_path)
    )
    tool = ObserveTool(env=env, image_store=image_store)

    with patch.object(image_store, "save") as mock_save:
        result = tool._run()

    # list 长度 1（无 image 块）
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["type"] == "text"

    # save 调用次数 == 0
    assert mock_save.call_count == 0


# ============================================================================
# 1.2.4: image_store=None 但 rgb 有值时降级
# ============================================================================


def test_observe_tool_degrades_without_image_store():
    """image_store=None + rgb 有值 → 行为与场景 1 一致（list 长度 1）。"""
    rgb = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    env = FakeEnv(_default_obs(rgb=rgb))
    tool = ObserveTool(env=env)  # image_store=None

    result = tool._run()

    # 与场景 1 行为一致
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert "末端执行器位置" in result[0]["text"]
    # 无 image 块
    assert not any(b.get("type") == "image" for b in result)


# ============================================================================
# 1.2.5: recorder 仍能落盘 + image_store 记录 2 个 URL
# ============================================================================


def test_observe_tool_recorder_emit_still_works_with_image_store(tmp_path: Path):
    """注入 ImageStore + 启用全局 recorder → observer/000/001.png + 2 URL。"""
    from experiment.recorder import (
        ExperimentRecorder,
        get_recorder,
        set_recorder,
    )

    # 重置全局 recorder，再注入新实例
    set_recorder(None)
    recorder = ExperimentRecorder(
        root=tmp_path / "exp_root", log_to_stdout=False
    )
    set_recorder(recorder)
    exp_dir = recorder.start()
    try:
        rgb1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        rgb2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        image_store = ImageStore(
            MemoryBackend(max_memory_items=5, spill_dir=tmp_path / "img_spill")
        )
        # Mock upload_to_minimax 避免真打 MiniMax
        image_store.upload_to_minimax = MagicMock(side_effect=lambda url: f"mm_file://{url.split('/')[-1].replace('.png', '')}")

        # _call_count 是 ObserveTool 实例级计数器；同一 tool 调两次 → idx 0/1。
        tool = ObserveTool(env=FakeEnv(_default_obs(rgb=rgb1)), image_store=image_store)

        result1 = tool.invoke({})
        # 切换到第二个 rgb（重置 env 让下次 get_obs 返回新图）
        tool.env = FakeEnv(_default_obs(rgb=rgb2))
        result2 = tool.invoke({})

        # tool.invoke({}) 在 BaseTool 包装下返回 _run() 的原始结果（即 list[dict]）。
        # 若未来 LangChain 改为返回 ToolMessage，则从 .content 字段取 list。
        if hasattr(result1, "content") and isinstance(result1.content, list):
            content1 = result1.content
            content2 = result2.content
        else:
            content1 = result1
            content2 = result2

        # 验证返回结构
        assert isinstance(content1, list)
        assert len(content1) == 2
        assert isinstance(content2, list)
        assert len(content2) == 2

        # 验证 recorder 落盘
        assert (recorder.observer_dir / "000.png").exists()
        assert (recorder.observer_dir / "001.png").exists()

        # 验证 experiment.log 含 [observe] saved
        log_content = (recorder.exp_dir / "experiment.log").read_text(
            encoding="utf-8"
        )
        assert "[observe] saved observer/000.png" in log_content
        assert "[observe] saved observer/001.png" in log_content

        # 验证 image_store.list("observations") 含 2 个 URL
        urls = image_store.list("observations")
        assert len(urls) == 2
        for url in urls:
            assert url.startswith("img://observations/")
    finally:
        recorder.finish(success=True, summary="x")
        set_recorder(None)
        # 防止 set_recorder(None) 后下一个测试的 get_recorder 拿到 safe recorder
        # （下一次 ObserveTool 构造时会再调一次 get_recorder）
        _ = get_recorder()


# ============================================================================
# 1.2.6: MockBackend 记录 save 参数
# ============================================================================


class _MockBackend(StorageBackend):
    """Mock StorageBackend：记录 save 调用的 (image, category, filename) 参数。"""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, str, str]] = []

    def save(self, image: np.ndarray, category: str, filename: str) -> None:
        self.calls.append((image, category, filename))

    def load(self, category: str, filename: str) -> np.ndarray:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def exists(self, category: str, filename: str) -> bool:
        return True

    def list(self, category=None):
        return []


def test_observe_tool_mock_backend_records_save_params():
    """MockBackend 统计 save 调用，验证 category + filename 格式。

    upload_to_minimax 被 mock 掉，避免真打 MiniMax。
    """
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    env = FakeEnv(_default_obs(rgb=rgb))

    mock_backend = _MockBackend()
    image_store = ImageStore(mock_backend)
    # Mock upload_to_minimax 避免真打 MiniMax
    image_store.upload_to_minimax = MagicMock(return_value="mm_file://fake_id")
    tool = ObserveTool(env=env, image_store=image_store)

    result = tool._run()

    # save 被调 1 次
    assert len(mock_backend.calls) == 1
    saved_image, saved_category, saved_filename = mock_backend.calls[0]

    # category == "observations"
    assert saved_category == "observations"

    # filename 匹配 YYYYMMDD_HHMMSS_NNN.png
    assert re.match(r"^\d{8}_\d{6}_\d{3}\.png$", saved_filename), (
        f"filename {saved_filename!r} does not match expected pattern"
    )

    # 返回 list 长度 2，第二块是 image 块
    assert len(result) == 2
    assert result[1]["type"] == "image"
    assert result[1]["url"] == "mm_file://fake_id"
    image_store.upload_to_minimax.assert_called_once()


# ============================================================================
# 1.2.7: target 过滤 + image 块仍存在
# ============================================================================


def test_observe_tool_target_filter_with_image_store(tmp_path: Path):
    """target 过滤时 text 块只含 red_block，image 块仍存在。

    upload_to_minimax 被 mock 掉，避免真打 MiniMax。
    """
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    env = FakeEnv(_default_obs(rgb=rgb))
    image_store = ImageStore(
        MemoryBackend(max_memory_items=5, spill_dir=tmp_path)
    )
    # Mock upload_to_minimax 避免真打 MiniMax
    image_store.upload_to_minimax = MagicMock(return_value="mm_file://fake_id")
    tool = ObserveTool(env=env, image_store=image_store)

    result = tool._run(target="red_block")

    # list 长度 2
    assert len(result) == 2

    # text 块只含 red_block
    text = result[0]["text"]
    assert "red_block" in text
    assert "blue_box" not in text

    # image 块仍存在
    assert result[1]["type"] == "image"
    assert result[1]["url"] == "mm_file://fake_id"


# ============================================================================
# 1.2.8: text 块内容与 _format_text_lines 一致
# ============================================================================


def test_observe_tool_text_block_preserves_formatting():
    """text 块 text 字段去掉 list 包装后与 _format_text_lines 直接调用一致。"""
    rgb = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    obs = _default_obs(rgb=rgb)
    env = FakeEnv(obs)
    tool = ObserveTool(env=env)  # 无 image_store → 不会追加 image 块

    result = tool._run(target="red_block")

    # 拿到 text 块
    text_block = result[0]
    text_field = text_block["text"]

    # 拆成行
    text_lines = text_field.split("\n")

    # 直接调 _format_text_lines 作对照
    expected_lines = _format_text_lines(obs, target="red_block")
    assert text_lines == expected_lines
