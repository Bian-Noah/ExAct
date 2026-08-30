"""静态断言测试：防止 OpenVLA 接线遗漏回归（源码级检查，不运行脚本）。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_verify_full_link_passes_openvla_config():
    src = (PROJECT_ROOT / "src/pipeline/script/verify_full_link.py").read_text(encoding="utf-8")
    assert "openvla=cfg.vla.openvla" in src


def test_verify_three_stage_pick_passes_openvla_config():
    src = (PROJECT_ROOT / "src/pipeline/script/verify_three_stage_pick.py").read_text(encoding="utf-8")
    assert "openvla=cfg.vla.openvla" in src


def test_openvla_readme_not_placeholder():
    readme = (PROJECT_ROOT / "src/executor/model/openvla/README.md").read_text(encoding="utf-8")
    assert "占位" not in readme
    assert "没有任何实现代码" not in readme
    assert "requirements-openvla.txt" in readme
