"""verify_three_stage_pick.py 单元 + 冒烟测试。

分层：
  - 纯函数测试：parse_args / _parse_stages / _make_run_dir / _dump_model_output
  - 冒烟测试：用 JointMockVLA 跑完整 main()，验证 3 个 stage 都跑通
    （不依赖真实 VLA 权重 / GPU，可在普通 CI 跑）

运行：
    PYTHONPATH=src python -m pytest tests/scripts/test_verify_three_stage_pick.py -v
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "src" / "pipeline" / "script" / "verify_three_stage_pick.py"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 把脚本所在目录加入 sys.path，让 importlib 能加载模块名 "verify_three_stage_pick"
SCRIPT_DIR = SCRIPT_PATH.resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


@pytest.fixture(scope="module")
def script_mod():
    """延迟导入被测脚本模块。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "verify_three_stage_pick", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# 纯函数测试
# ============================================================

class TestParseStages:
    """_parse_stages 把 CLI 字符串解析为 (label, instruction) 元组列表。"""

    def test_default_returns_three_stages(self, script_mod):
        """--stages 为空 → 返回默认 3 阶段。"""
        stages = script_mod._parse_stages(None)
        assert len(stages) == 3
        labels = [s[0] for s in stages]
        instrs = [s[1] for s in stages]
        assert labels == ["approach", "descend", "grasp"]
        # 每条指令必须符合 smolVLA 规范：动词开头 + 全英文 + ≤30 字符
        for ins in instrs:
            assert ins[0].isalpha(), f"instruction should start with letter: {ins!r}"
            assert len(ins) <= 30, f"instruction too long: {ins!r}"

    def test_colon_separator(self, script_mod):
        """'approach:move above cube,grasp:close gripper' → 2 条。"""
        stages = script_mod._parse_stages("approach:move above cube,grasp:close gripper")
        assert stages == (
            ("approach", "move above cube"),
            ("grasp", "close gripper"),
        )

    def test_equals_separator_with_spaces(self, script_mod):
        """兼容 'label = instruction' 写法。"""
        stages = script_mod._parse_stages("foo = bar baz, qux = quux")
        assert stages == (("foo", "bar baz"), ("qux", "quux"))

    def test_label_sanitized(self, script_mod):
        """label 中的非 [a-zA-Z0-9_] 字符被替换为 '_'。"""
        stages = script_mod._parse_stages("foo-bar.baz:do thing")
        assert stages[0][0] == "foo_bar_baz"

    def test_empty_raises(self, script_mod):
        """全部空段 / 无 segment → ValueError。"""
        with pytest.raises(ValueError, match="--stages"):
            script_mod._parse_stages("")
        with pytest.raises(ValueError, match="--stages"):
            script_mod._parse_stages(",,,")

    def test_missing_separator_raises(self, script_mod):
        """缺少 label:instruction 分隔符 → ValueError。"""
        with pytest.raises(ValueError, match="分隔符"):
            script_mod._parse_stages("just_a_word")


class TestMakeRunDir:
    """_make_run_dir 创建 {prefix}_{timestamp}/ 目录，区别于 verify_full_link。"""

    def test_uses_prefix(self, script_mod, tmp_path):
        """run_dir 名称必须含 prefix。"""
        run_dir = script_mod._make_run_dir(tmp_path, "three_stage_pick")
        assert run_dir.exists()
        assert run_dir.parent == tmp_path
        assert run_dir.name.startswith("three_stage_pick_")

    def test_collision_appends_suffix(self, script_mod, tmp_path):
        """同秒重复创建 → 自动追加 _001 后缀。"""
        first = script_mod._make_run_dir(tmp_path, "test_prefix")
        second = script_mod._make_run_dir(tmp_path, "test_prefix")
        assert first != second
        assert first.name.startswith("test_prefix_")
        assert second.name.startswith("test_prefix_")

    def test_distinct_from_verify_full_link(self, script_mod, tmp_path):
        """用与 verify_full_link 不同前缀，目录名不冲突。"""
        run_dir = script_mod._make_run_dir(tmp_path, "three_stage_pick")
        # verify_full_link 用的是纯 timestamp 命名（无前缀），前缀化后两者可区分
        assert "three_stage_pick" in run_dir.name


class TestDumpModelOutput:
    """_dump_model_output 把原始 VLAOutput.values 完整 dump 到文本文件。"""

    def test_writes_expected_sections(self, script_mod, tmp_path):
        """文件应含 shape/dtype/各列统计/完整 chunk 行。"""
        from env.base import ActionSpec
        spec = ActionSpec("joint", ("joint",) * 6)
        values = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            ],
            dtype=np.float32,
        )
        path = script_mod._dump_model_output(tmp_path, "approach", values, spec)
        text = Path(path).read_text()

        # header
        assert "# raw VLAOutput.values dump (BEFORE adapter)" in text
        assert "# shape: (2, 6)" in text
        assert "# dtype: float32" in text
        assert "# spec.space: joint" in text
        # 统计
        assert "# per-column min:" in text
        assert "# per-column max:" in text
        assert "# per-column mean:" in text
        assert "# per-column std:" in text
        # 完整 chunk 行
        assert "# --- full chunk ---" in text
        assert "0.100000,0.200000,0.300000,0.400000,0.500000,0.600000" in text
        assert "0.700000,0.800000,0.900000,1.000000,1.100000,1.200000" in text

    def test_handles_1d_input(self, script_mod, tmp_path):
        """1D 输入也接受（reshape 到 (1, -1) 后写入）。"""
        from env.base import ActionSpec
        spec = ActionSpec("joint", ("joint",) * 4)
        values = np.array([0.1, 0.2, 0.3, 0.4])
        path = script_mod._dump_model_output(tmp_path, "s1", values, spec)
        text = Path(path).read_text()
        assert "# shape: (1, 4)" in text
        assert "0.100000,0.200000,0.300000,0.400000" in text


class TestConstants:
    """默认配置的稳定性。"""

    def test_run_dir_prefix(self, script_mod):
        """prefix 默认值应为 'three_stage_pick'（区别于 verify_full_link）。"""
        assert script_mod.RUN_DIR_PREFIX == "three_stage_pick"

    def test_default_stages_count(self, script_mod):
        """默认 3 个 stage：approach / descend / grasp。"""
        assert len(script_mod.DEFAULT_STAGES) == 3
        labels = tuple(s[0] for s in script_mod.DEFAULT_STAGES)
        assert labels == ("approach", "descend", "grasp")


# ============================================================
# 冒烟测试：用 JointMockVLA 跑完整 main() 一次
# ============================================================

@pytest.mark.slow
class TestMainSmoke:
    """用 JointMockVLA 跑一遍 main()，确认 3 个 stage 全跑通、文件全产出。

    依赖：env.pybullet_env + executor.model.mock.JointMockVLA
    （这两个在 Mac / Linux 无 torch 也能跑）。
    """

    def test_main_runs_three_stages_with_mock(self, script_mod, tmp_path,
                                              monkeypatch):
        """端到端：用 --vla mock 跑完整三阶段，断言产物文件。"""
        # 把 SCRIPT_DATA_ROOT 改到 tmp_path，避免污染项目 data 目录
        monkeypatch.setattr(script_mod, "SCRIPT_DATA_ROOT", tmp_path)

        # 调 main()：传 --vla mock --no-gui 走 JointMockVLA（适合 SO101 单臂）
        monkeypatch.setattr(
            sys, "argv",
            [
                "verify_three_stage_pick.py",
                "--vla", "mock",
                "--no-gui",
                "--data-root", str(tmp_path),
                "--prefix", "smoke",
            ],
        )

        # 默认指令走 smolVLA 规范，JointMockVLA 故意忽略 image，
        # 但要求 adapter 为 joint→joint（因为 JointMockVLA 输出 joint 空间 6 维）
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = script_mod.main()
        assert rc == 0, f"main() returned {rc}; stdout: {buf.getvalue()[-500:]}"

        # 验证产物目录
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        assert run_dir.name.startswith("smoke_")

        # experiment.log 必须存在且非空
        log_path = run_dir / "experiment.log"
        assert log_path.exists()
        log_text = log_path.read_text()
        assert "verify_three_stage_pick started" in log_text
        assert "verify_three_stage_pick finished" in log_text
        # 3 个 stage 标签都在 log 里
        for lbl in ("approach", "descend", "grasp"):
            assert f"label={lbl!r}" in log_text, (
                f"stage {lbl!r} missing from log"
            )

        # 3 个 model_output_stageN.txt 都产出
        for lbl in ("approach", "descend", "grasp"):
            dump = run_dir / f"model_output_{lbl}.txt"
            assert dump.exists(), f"missing dump file: {dump.name}"
            dump_text = dump.read_text()
            assert "# raw VLAOutput.values dump (BEFORE adapter)" in dump_text

        # 至少 4 张 step_XXX.png（reset + 3 stage end）
        pngs = sorted(run_dir.glob("step_*.png"))
        assert len(pngs) >= 4, f"expected ≥4 step pngs, got {len(pngs)}"

        # input/ 目录含 3 个 stage 的多视角输入图
        input_dir = run_dir / "input"
        assert input_dir.is_dir()
        # 每个 stage 至少 1 张图（默认 1 相机；3 路相机配置下 3 张）
        for lbl in ("approach", "descend", "grasp"):
            stage_imgs = list(input_dir.glob(f"{lbl}_chunk00_*.png"))
            assert stage_imgs, f"no input images for stage {lbl!r}"
