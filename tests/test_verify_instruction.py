"""validate_instruction 单元测试（English-only 版）。

覆盖：英文合规 / 英文边界 / 非英文拒绝 / 空 / 句末标点 / 长度。
"""

from __future__ import annotations

from utils.verify import (
    MAX_INSTRUCTION_LEN,
    SENTENCE_SEPARATORS,
    VERB_PREFIXES,
    validate_instruction,
)


# ========== 正常输入（合规英文指令） ==========


class TestValidInstructions:
    def test_pick_cube(self):
        assert validate_instruction("pick red cube") is None

    def test_grab_cube(self):
        assert validate_instruction("grab the red block") is None

    def test_move_with_coords(self):
        assert validate_instruction("move to (0.5, 0, 0.3)") is None

    def test_move_with_x_y_z(self):
        assert validate_instruction("move to x=0.5, y=0.0, z=0.4") is None

    def test_minimal(self):
        # "p" 单独不构成动词原形前缀 → 拒绝
        # 最小合规应是 "place"（5 字符）
        assert validate_instruction("p") == "不是动作动词开头"
        assert validate_instruction("place") is None

    def test_push(self):
        assert validate_instruction("push") is None

    def test_rotate(self):
        assert validate_instruction("rotate left") is None

    def test_lift_up(self):
        assert validate_instruction("lift up") is None

    def test_turn_right(self):
        assert validate_instruction("turn right") is None


# ========== 异常输入：中文拒绝 ==========


class TestChineseRejected:
    def test_chinese_action_rejected(self):
        # 中文动作指令一律拒绝
        assert validate_instruction("夹取红色方块") == "包含非英文字符（指令必须全英文）"

    def test_chinese_move_rejected(self):
        assert validate_instruction("移动到红色方块上方") == "包含非英文字符（指令必须全英文）"

    def test_mixed_chinese_english_rejected(self):
        # 中英混排也拒绝
        assert validate_instruction("pick 红色方块") == "包含非英文字符（指令必须全英文）"

    def test_chinese_punctuation_rejected(self):
        # 全角中文标点拒绝
        assert validate_instruction("pick，red cube") == "包含非英文字符（指令必须全英文）"


# ========== 异常输入：空 ==========


class TestEmptyInstructions:
    def test_empty_string(self):
        assert validate_instruction("") == "指令为空"

    def test_whitespace_only(self):
        assert validate_instruction("   ") == "指令为空"

    def test_newline_only(self):
        # "\n" 在 SENTENCE_SEPARATORS，但空检查先命中
        assert validate_instruction("\n") == "指令为空"


# ========== 异常输入：动词开头（英文） ==========


class TestVerbPrefix:
    def test_noun_start_rejected(self):
        # "red cube" 首词 "red" 不是动词原形
        assert validate_instruction("red cube") == "不是动作动词开头"

    def test_x_start_rejected(self):
        # "xyz" 不是任何动词原形的前缀
        assert validate_instruction("xyz some action") == "不是动作动词开头"

    def test_q_start_rejected(self):
        # "quit" 不是白名单中的动词
        assert validate_instruction("quit the task") == "不是动作动词开头"

    def test_uppercase_verb_accepted(self):
        # 大小写不敏感："Pick" 仍以 "pick" 开头
        assert validate_instruction("Pick red cube") is None

    def test_digit_start_rejected(self):
        # 数字开头不是动词
        assert validate_instruction("1st place") == "不是动作动词开头"

    def test_partial_verb_match_rejected(self):
        # "picker" 不是 "pick" 的原形（pick 是动词，picker 是名词）
        assert validate_instruction("picker of cubes") == "不是动作动词开头"

    def test_verb_followed_by_noun(self):
        # "rotate cube" → rotate 是动词原形 → 合规
        assert validate_instruction("rotate cube") is None

    def test_verb_followed_by_prep(self):
        assert validate_instruction("move to x=0.5") is None


# ========== 异常输入：多句 ==========


class TestSingleSentence:
    def test_english_period(self):
        assert validate_instruction("pick red cube.") == "包含多个句子"

    def test_question_mark(self):
        assert validate_instruction("pick? red cube") == "包含多个句子"

    def test_exclamation(self):
        assert validate_instruction("pick! red cube") == "包含多个句子"

    def test_newline(self):
        assert validate_instruction("pick\nred cube") == "包含多个句子"

    def test_chinese_period(self):
        # 中文句号同样视为多句
        assert validate_instruction("pick red cube。") == "包含非英文字符（指令必须全英文）"


# ========== 异常输入：长度 ==========


class TestLength:
    def test_over_limit(self):
        # 必须是英文动词开头
        s = "place " + "a" * 30  # 36 字符
        assert validate_instruction(s) == "超过 30 字符上限"

    def test_at_limit_with_verb(self):
        # 30 字符以动词原形开头
        s = "place " + "a" * 24  # "place " (6) + 24 = 30
        assert len(s) == MAX_INSTRUCTION_LEN
        assert validate_instruction(s) is None

    def test_at_limit_plus_one(self):
        s = "place " + "a" * 25  # 31 字符
        assert len(s) == MAX_INSTRUCTION_LEN + 1
        assert validate_instruction(s) == "超过 30 字符上限"


# ========== 校验顺序 ==========


class TestCheckOrder:
    def test_empty_takes_priority(self):
        assert validate_instruction("") == "指令为空"

    def test_non_english_takes_priority_over_verb(self):
        # 中文 + 看似动词开头 → 优先报告非英文
        assert validate_instruction("夹") == "包含非英文字符（指令必须全英文）"

    def test_non_english_takes_priority_over_length(self):
        s = "夹" + "a" * 30
        assert validate_instruction(s) == "包含非英文字符（指令必须全英文）"

    def test_verb_takes_priority_over_length(self):
        # 英文非动词 + 超长 → 优先报告动词
        assert validate_instruction("x" * 100) == "不是动作动词开头"

    def test_sentence_takes_priority_over_length(self):
        s = "place " + ("a. " * 20)  # 含多个 "."，超长
        assert validate_instruction(s) == "包含多个句子"


# ========== 小数点豁免 ==========


class TestDecimalNotCounted:
    def test_decimal_in_coords(self):
        assert validate_instruction("move to 0.5") is None

    def test_decimal_in_xyz_format(self):
        assert validate_instruction("move to x=0.5") is None


# ========== 纯函数性质 ==========


class TestPureFunction:
    def test_does_not_mutate_input(self):
        original = "pick"
        validate_instruction(original)
        assert original == "pick"

    def test_returns_new_value(self):
        result = validate_instruction("pick")
        assert result is None


# ========== 常量导出 ==========


class TestConstants:
    def test_max_len_is_30(self):
        assert MAX_INSTRUCTION_LEN == 30

    def test_verb_prefixes_are_lowercase_english(self):
        for p in VERB_PREFIXES:
            assert p.isascii()
            assert p.isalpha()
            assert p.islower()

    def test_sentence_separators(self):
        assert "?" in SENTENCE_SEPARATORS
        assert "\n" in SENTENCE_SEPARATORS
        assert "!" in SENTENCE_SEPARATORS
        # "." 已从小数处理分离
        assert "." not in SENTENCE_SEPARATORS