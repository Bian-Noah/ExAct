"""smolVLA 指令校验：英文动词开头 / 全英文 / 单句 / ≤30 字符。

iter9-verify-explore-design 引入：纯函数 `validate_instruction`，无 LLM/env 依赖，
返回不合规原因或 None。在 ActionTool._run 入口拦截。

iter9-extend（English-only）：smolVLA 训练语料以英文为主，强制要求指令
全英文（仅 ASCII 字符），以白名单中的动词原形开头。
"""

from __future__ import annotations

from typing import Optional

# 英文动词原形白名单（用于前缀匹配）：覆盖 smolVLA 训练语料高频动作。
# 必须在 prefix 维度上以这些原形之一开头（大小写不敏感）。
# pick / grab / place / push / pull / press / turn / rotate / open / close / move / look / touch / lift / drop / insert
# 方向词（与 move/go 组合）：forward / backward / up / down / left / right
VERB_PREFIXES: tuple[str, ...] = (
    "pick", "grab", "place", "push", "pull", "press", "turn", "rotate",
    "open", "close", "move", "look", "touch", "lift", "drop", "insert",
    "forward", "backward", "up", "down", "left", "right",
)
# 加速首字符判断（按首字母小写分组）
_VERB_FIRST_CHARS: frozenset[str] = frozenset(p[0] for p in VERB_PREFIXES)

# 句末标点：出现任一即视为多句
SENTENCE_SEPARATORS: frozenset[str] = frozenset(
    {"?", "!", "\n", "。", "？", "！"}
)

# 字符上限：smolVLA 训练时按 token 数截断
MAX_INSTRUCTION_LEN: int = 30


def _has_non_english_chars(s: str) -> bool:
    """检测字符串中是否有非英文字符（CJK / 全角 / 其他非 ASCII）。"""
    for c in s:
        if c.isascii():
            continue
        return True
    return False


def _starts_with_verb(s: str) -> bool:
    """大小写不敏感地判断 s 是否以白名单中的动词原形开头。"""
    lower = s.lower()
    first = lower[0]
    if first not in _VERB_FIRST_CHARS:
        return False
    return any(lower.startswith(prefix) for prefix in VERB_PREFIXES)


def validate_instruction(instruction: str) -> Optional[str]:
    """校验指令是否符合 smolVLA 规范（全英文版）。

    校验顺序：空检查 → 全英文 → 动词开头 → 单句 → 长度。任一不通过即返回原因，
    全部通过返回 None。

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

    # 动词前缀匹配（大小写不敏感）+ 单词边界（避免 "picker" 误匹配 "pick"）
    lower = s.lower()
    first = lower[0]
    if first not in _VERB_FIRST_CHARS:
        return "不是动作动词开头"
    matched = False
    for prefix in VERB_PREFIXES:
        if lower.startswith(prefix):
            # 边界检查：动词原形后必须是空格 / 字符串末尾 / 非字母字符
            end = len(prefix)
            if end == len(s) or s[end] == " " or not s[end].isalpha():
                matched = True
                break
    if not matched:
        return "不是动作动词开头"

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
            return "包含多个句子"

    # 长度上限（按字符数）
    if len(s) > MAX_INSTRUCTION_LEN:
        return "超过 30 字符上限"

    return None