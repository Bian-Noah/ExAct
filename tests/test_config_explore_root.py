"""ExploreConfig.root 字段加载测试。
"""

from __future__ import annotations

import textwrap

import pytest

from config.loader import ExploreConfig, load_config


def test_default_root():
    cfg = ExploreConfig()
    assert cfg.root == "data/explore"
    assert cfg.enabled is False


def test_explicit_root():
    cfg = ExploreConfig(root="/tmp/custom")
    assert cfg.root == "/tmp/custom"


def test_load_yaml_default(tmp_path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text(textwrap.dedent("""
        env: {}
        vla: {}
        llm: {}
        explore:
          enabled: true
        robot: {}
        task: {}
        agent: {}
        experiment: {}
        image_store: {}
    """).strip(), encoding="utf-8")
    cfg = load_config(str(yaml))
    assert cfg.explore.enabled is True
    assert cfg.explore.root == "data/explore"  # 默认值


def test_load_yaml_explicit_root(tmp_path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text(textwrap.dedent("""
        env: {}
        vla: {}
        llm: {}
        explore:
          enabled: true
          root: "/tmp/myexplore"
        robot: {}
        task: {}
        agent: {}
        experiment: {}
        image_store: {}
    """).strip(), encoding="utf-8")
    cfg = load_config(str(yaml))
    assert cfg.explore.root == "/tmp/myexplore"


def test_load_yaml_root_type_error(tmp_path):
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text(textwrap.dedent("""
        env: {}
        vla: {}
        llm: {}
        explore:
          enabled: true
          root: 123
        robot: {}
        task: {}
        agent: {}
        experiment: {}
        image_store: {}
    """).strip(), encoding="utf-8")
    with pytest.raises(TypeError, match="ExploreConfig.root"):
        load_config(str(yaml))


def test_default_yaml_has_root(tmp_path):
    """项目 configs/default.yaml 必须含 explore.root 字段。"""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(project_root, "configs", "default.yaml"))
    assert cfg.explore.root == "data/explore"