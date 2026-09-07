"""OpenVLA bridge_orig 指令校验：非空 / 全英文 / 单句 / 长度 ≤ 50。

iter9-verify-explore-design 引入：纯函数 `validate_instruction`，无 LLM/env 依赖，
返回不合规原因或 None。在 ActionTool._run 入口拦截。

iter9-extend（English-only）：OpenVLA 训练语料以英文为主，强制要求指令
全英文（仅 ASCII 字符），单句，且长度不超过 50 字符。

不做严格词汇限制：instruction 是被拼到
"What action should the robot take to {instruction}?" 喂给 OpenVLA 的，
只要贴合 bridge_orig 训练分布（动词+宾语+可选位置/容器）即可；
硬限制动词白名单会让 LLM 在合理同义词上被误拒（grasp/lower/descend 等）。
"""

from __future__ import annotations

from typing import Optional

# 句末标点：出现任一即视为多句
SENTENCE_SEPARATORS: frozenset[str] = frozenset(
    {"?", "!", "\n", "。", "？", "！"}
)

# 字符上限：OpenVLA bridge_orig prompt 模板预留 50 字符余量给 instruction
MAX_INSTRUCTION_LEN: int = 50


def _has_non_english_chars(s: str) -> bool:
    """检测字符串中是否有非英文字符（CJK / 全角 / 其他非 ASCII）。"""
    for c in s:
        if c.isascii():
            continue
        return True
    return False


def validate_instruction(instruction: str) -> Optional[str]:
    """校验指令是否符合 OpenVLA bridge_orig 训练分布的格式约束。

    校验顺序：空检查 → 全英文 → 单句 → 长度。任一不通过即返回原因，
    全部通过返回 None。

    不查动词白名单、首字母筛、单词边界：这些约束过严会误拒合理同义词
    （grasp/reach/lower/descend 等），影响 LLM 表达空间。指令是否贴合训练分布
    由 LLM 自己在 extra_prompt 指引下保证，本校验只做"格式层面"拦截。

    Args:
        instruction: LLM 下发给 ActionTool 的指令字符串。

    Returns:
        None（合规）或不合规原因字符串。
    """
    if not instruction or not instruction.strip():
        return "指令为空"

    s = instruction.strip()

    # 全英文校验：含中文/全角/其他非 ASCII 字符则拒绝
    if _has_non_english_chars(s):
        return "包含非英文字符（指令必须全英文）"

    # 单句：不含句末标点；英文 "." 仅在前后都是数字时算小数点
    for i, c in enumerate(s):
        if c == ".":
            prev_digit = i > 0 and s[i - 1].isdigit()
            next_digit = i + 1 < len(s) and s[i + 1].isdigit()
            if prev_digit and next_digit:
                continue  # 小数点内部（如 0.5），跳过
            # 否则视为句末标点
            return "包含多个句子"
        if c in SENTENCE_SEPARATORS:
            return "包含多个句子或问号/感叹号"

    # 长度上限（按字符数）
    if len(s) > MAX_INSTRUCTION_LEN:
        return f"超过 {MAX_INSTRUCTION_LEN} 字符上限"

    return None