"""A fixed-cell terminal composition shared by live display and recorded exports.

Localised to Simplified Chinese. Because CJK glyphs occupy two terminal columns
while box-drawing and block glyphs occupy one, the canvas tracks *display
columns* rather than raw character counts: ``put`` advances by the display width
of each character and marks the trailing cell of a wide glyph with an empty
string, which contributes nothing when the row is joined.
"""

from rich.text import Text

BG = "#090f13"
FG = "#e3f3ef"
MUTED = "#68868c"
DIM = "#20353c"
GREEN = "#62f5b5"
AMBER = "#ffce73"
RED = "#ff7c8c"
CYAN = "#8ad8e9"

DIGITS = {
    "0": ("█▀█", "█ █", "▀▀▀"),
    "1": ("▄█ ", " █ ", "▀▀▀"),
    "2": ("▀▀█", "█▀▀", "▀▀▀"),
    "3": ("▀▀█", "▀▀█", "▀▀▀"),
    "4": ("█ █", "▀▀█", "  ▀"),
    "5": ("█▀▀", "▀▀█", "▀▀▀"),
    "6": ("█▀▀", "█▀█", "▀▀▀"),
    "7": ("▀▀█", "  █", "  ▀"),
    "8": ("█▀█", "█▀█", "▀▀▀"),
    "9": ("█▀█", "▀▀█", "▀▀▀"),
}

DIRECTION_ZH = {"UP": "上", "DOWN": "下", "LEFT": "左", "RIGHT": "右"}


def char_width(character):
    """Terminal display width of a single character (1 or 2 columns)."""
    code = ord(character)
    if code < 0x1100:
        return 1
    if (
        0x1100 <= code <= 0x115F  # Hangul Jamo
        or 0x2E80 <= code <= 0x303E  # CJK radicals, Kangxi, CJK punctuation
        or 0x3041 <= code <= 0x33FF  # kana, CJK compatibility
        or 0x3400 <= code <= 0x4DBF  # CJK extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK unified ideographs
        or 0xA000 <= code <= 0xA4CF  # Yi
        or 0xAC00 <= code <= 0xD7A3  # Hangul syllables
        or 0xF900 <= code <= 0xFAFF  # CJK compatibility ideographs
        or 0xFE30 <= code <= 0xFE6F  # CJK compatibility forms
        or 0xFF00 <= code <= 0xFF60  # fullwidth forms
        or 0xFFE0 <= code <= 0xFFE6  # fullwidth signs
        or 0x1F300 <= code <= 0x1FAFF  # emoji
        or 0x20000 <= code <= 0x3FFFD  # CJK extensions B+
    ):
        return 2
    return 1


def display_width(text):
    return sum(char_width(character) for character in str(text))


def pad(text, columns):
    """Right-pad to a display width, counting wide glyphs as two columns."""
    text = str(text)
    return text + " " * max(0, columns - display_width(text))


class Canvas:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.chars = [[" "] * width for _ in range(height)]
        self.styles = [[FG] * width for _ in range(height)]

    def put(self, row, column, text, color=FG):
        if not 0 <= row < self.height:
            return
        x = column
        for character in str(text):
            span = char_width(character)
            if 0 <= x < self.width:
                self.chars[row][x], self.styles[row][x] = character, color
                if span == 2 and x + 1 < self.width:
                    # Trailing cell of a wide glyph: nothing of its own to render.
                    self.chars[row][x + 1], self.styles[row][x + 1] = "", color
            x += span

    def bar(self, row, column, value, length=20, color=GREEN):
        count = round(max(0, min(1, value)) * length)
        self.put(row, column, "━" * length, DIM)
        self.put(row, column, "━" * count, color)

    def number(self, row, column, value, color=GREEN):
        digits = f"{value:03d}"
        for index, digit in enumerate(digits):
            for line, glyphs in enumerate(DIGITS[digit]):
                self.put(row + line, column + index * 4, glyphs, color)

    def rich_text(self):
        output = Text(no_wrap=True, overflow="crop", style=f"{FG} on {BG}")
        for row, (chars, styles) in enumerate(zip(self.chars, self.styles)):
            begin = 0
            for end in range(1, self.width + 1):
                if end == self.width or styles[end] != styles[begin]:
                    output.append("".join(chars[begin:end]), style=styles[begin])
                    begin = end
            if row + 1 != self.height:
                output.append("\n")
        return output


def layout_size(width, height):
    return max(104, width * 2 + 50), max(35, height + 19)


def compose(game, decision, stats):
    """Probabilities describe the displayed board, before its announced next step."""
    width, height = layout_size(game["width"], game["height"])
    c = Canvas(width, height)
    left, right, top = 3, max(58, game["width"] * 2 + 10), 6
    side = width - right - 4
    bottom = top + game["height"] + 1
    state = (
        "已暂停"
        if stats.get("paused")
        else ("棋盘清空" if game["won"] else "游戏结束" if not game["alive"] else "运行中")
    )
    if stats.get("replay") and state == "运行中":
        state = "回放 · 1×"
    c.put(1, left, "LAYA  /  本地智能", MUTED)
    c.put(1, width - display_width(state) - 3, state, GREEN if game["alive"] else RED)
    c.put(2, left, "─" * (width - 6), DIM)
    c.put(4, left, "贪 吃 蛇", FG)
    c.put(4, left + 31, f"第 {stats.get('round', 1):02d} 局", MUTED)
    c.put(top, left, "┌" + "─" * (game["width"] * 2) + "┐", DIM)
    c.put(bottom, left, "└" + "─" * (game["width"] * 2) + "┘", DIM)
    for y in range(game["height"]):
        c.put(top + 1 + y, left, "│", DIM)
        c.put(top + 1 + y, left + game["width"] * 2 + 1, "│", DIM)
        c.put(top + 1 + y, left + 1, "· " * game["width"], "#13272e")
    body = game["body"]
    for index, (x, y) in reversed(list(enumerate(body))):
        fraction = 1 - index / max(1, len(body))
        color = (
            "#dcfff0"
            if index == 0
            else (
                f"#{int(18 + 64 * fraction):02x}{int(73 + 150 * fraction):02x}"
                f"{int(57 + 102 * fraction):02x}"
            )
        )
        c.put(top + y + 1, left + 1 + 2 * x, "██", color)
    if game["food"] is not None:
        x, y = game["food"]
        c.put(top + y + 1, left + 1 + 2 * x, "● ", AMBER)
    for offset, label, value, color in (
        (0, "得分", game["score"], GREEN),
        (18, "长度", game["length"], FG),
        (36, "最佳", stats.get("best", game["score"]), MUTED),
    ):
        c.put(bottom + 2, left + offset, label, MUTED)
        c.number(bottom + 3, left + offset, value, color)
    c.bar(bottom + 7, left, game["length"] / (game["width"] * game["height"]), 41)
    c.put(
        bottom + 7,
        left + 43,
        f"{100 * game['length'] / (game['width'] * game['height']):4.1f}%",
        MUTED,
    )

    c.put(4, right, "Laya MLX", GREEN)
    c.put(5, right, f"{stats.get('hardware', 'Apple silicon')} · 本地", MUTED)
    c.put(7, right, "下一步", FG)
    c.put(7, right + 15, "模型概率", MUTED)
    probabilities = decision.get("probabilities", {})
    for index, direction in enumerate(("UP", "DOWN", "LEFT", "RIGHT")):
        row = 9 + index
        probability = probabilities.get(direction, 0)
        selected = direction == decision.get("proposed")
        color = GREEN if selected else MUTED
        c.put(row, right, f"{'›' if selected else ' '} {pad(DIRECTION_ZH[direction], 4)}", color)
        count = round(probability * 18)
        c.put(row, right + 9, "░" * 18, DIM)
        c.put(row, right + 9, "█" * count, color)
        c.put(row, right + 29, f"{probability:.2f}", color)
    c.put(14, right, "执行", MUTED)
    c.put(14, right + 12, DIRECTION_ZH.get(decision.get("executed"), "—"), GREEN)
    if decision.get("intervened"):
        c.put(14, right + 20, "护盾介入", AMBER)
    c.put(16, right, "死路风险", MUTED)
    risk = decision.get("dead_end_risk", 0)
    c.bar(17, right, risk, min(24, side - 9), AMBER if risk < 0.5 else RED)
    c.put(17, right + 29, f"{risk:.2f}", AMBER if risk < 0.5 else RED)
    c.put(19, right, "食物可达", MUTED)
    c.bar(20, right, decision.get("food_reachable", 0), min(24, side - 9), CYAN)
    c.put(20, right + 29, f"{decision.get('food_reachable', 0):.2f}", CYAN)
    c.put(22, right, "推理耗时", MUTED)
    c.put(22, right + 18, f"{decision.get('inference_ms', 0):5.1f} ms", FG)
    c.put(23, right, "决策速度", MUTED)
    c.put(23, right + 18, f"{stats.get('steps_per_second', 0):5.1f} /s", FG)
    c.put(24, right, "输出 Token", MUTED)
    c.put(24, right + 18, str(decision.get("output_tokens", 0)), FG)
    c.put(25, right, "网络", MUTED)
    c.put(25, right + 18, "离线", GREEN)
    c.put(26, right, "引擎", MUTED)
    c.put(26, right + 18, "MLX · FP16", MUTED)
    guarded = stats.get("guarded", True)
    c.put(28, right, "Laya + 循环安全" if guarded else "Laya · 护盾关闭", MUTED)
    c.put(29, right, f"护盾介入  {stats.get('interventions', 0):04d}", AMBER)
    c.put(height - 3, left, "─" * (width - 6), DIM)
    c.put(height - 2, left, "SPACE 暂停   ↑/↓ 调速   R 重开   Q 退出", MUTED)
    elapsed = stats.get("elapsed", 0)
    clock = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
    c.put(height - 2, right, f"由 LAYA 实时决策       {clock}", MUTED)
    return c
