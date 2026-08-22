"""DEFAULT_SYSTEM_PROMPT 单元测试（iter11-reset-multicam）。

验证 action 工具描述含 operation 字段说明,reset 路径语义被描述。
"""
from agents.prompt import DEFAULT_SYSTEM_PROMPT


def test_default_system_prompt_mentions_operation_field():
    """DEFAULT_SYSTEM_PROMPT 描述 action 工具 operation 字段。"""
    assert "operation" in DEFAULT_SYSTEM_PROMPT


def test_default_system_prompt_mentions_reset_operation():
    """DEFAULT_SYSTEM_PROMPT 描述 operation='reset' 语义。"""
    assert "operation='reset'" in DEFAULT_SYSTEM_PROMPT or 'operation="reset"' in DEFAULT_SYSTEM_PROMPT


def test_default_system_prompt_explains_reset_bypasses_vla():
    """DEFAULT_SYSTEM_PROMPT 说明 reset 不走 VLA 推理。"""
    # 搜索 reset 段落
    reset_section_start = DEFAULT_SYSTEM_PROMPT.find("operation='reset'")
    if reset_section_start == -1:
        reset_section_start = DEFAULT_SYSTEM_PROMPT.find('operation="reset"')
    assert reset_section_start > 0
    # 取 reset 段后 200 字符
    reset_section = DEFAULT_SYSTEM_PROMPT[reset_section_start: reset_section_start + 200]
    assert "不走 VLA" in reset_section or "bypass" in reset_section.lower()


def test_default_system_prompt_explains_reset_use_case():
    """DEFAULT_SYSTEM_PROMPT 描述 reset 使用时机(关节极限卡死时)。"""
    # reset 段落应包含使用场景描述
    assert "关节极限" in DEFAULT_SYSTEM_PROMPT or "卡在" in DEFAULT_SYSTEM_PROMPT


def test_default_system_prompt_still_has_smovla_validation():
    """DEFAULT_SYSTEM_PROMPT 仍保留 smolVLA 指令规范(动词开头/英文/单句/≤30 字符)。"""
    assert "smolVLA" in DEFAULT_SYSTEM_PROMPT
    assert "≤30 字符" in DEFAULT_SYSTEM_PROMPT or "30 字符" in DEFAULT_SYSTEM_PROMPT
    assert "动作动词" in DEFAULT_SYSTEM_PROMPT or "动词原形" in DEFAULT_SYSTEM_PROMPT


def test_default_system_prompt_mentions_multi_camera_observe():
    """DEFAULT_SYSTEM_PROMPT 提示 observe 可能返回多张图(多相机)。"""
    assert "多相机" in DEFAULT_SYSTEM_PROMPT or "多张" in DEFAULT_SYSTEM_PROMPT or "多个相机" in DEFAULT_SYSTEM_PROMPT


def test_prompt_requires_declaring_multi_view_image_count():
    """最终回答必须声明看到的视角数量(多视角图片)。"""
    assert "看到了" in DEFAULT_SYSTEM_PROMPT
    assert "个视角" in DEFAULT_SYSTEM_PROMPT
    assert "单视角" in DEFAULT_SYSTEM_PROMPT


def test_prompt_requires_declaring_reset_tool_usable():
    """最终回答必须声明复位工具是否正常可用。"""
    assert "复位工具" in DEFAULT_SYSTEM_PROMPT
    assert "正常使用" in DEFAULT_SYSTEM_PROMPT
    assert "不可用" in DEFAULT_SYSTEM_PROMPT


def test_prompt_anti_hallucination_for_views_and_reset():
    """prompt 明确禁止夸大视角数量与谎称复位成功。"""
    assert "不要假装看到了图像" in DEFAULT_SYSTEM_PROMPT
    assert "不要夸大" in DEFAULT_SYSTEM_PROMPT
    assert "不要谎称成功使用了复位工具" in DEFAULT_SYSTEM_PROMPT