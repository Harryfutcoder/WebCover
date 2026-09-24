import difflib
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Any

from action.web_action import WebAction
from state.web_state import WebState


class TagTable:
    TAG_MAP = {
        "a": 2,
        "/a": 3,
        "abbr": 4,
        "/abbr": 5,
        "acronym": 6,
        "/acronym": 7,
        "address": 8,
        "/address": 9,
        "applet": 10,
        "/applet": 11,
        "area": 12,
        "/area": 13,
        "article": 14,
        "/article": 15,
        "aside": 16,
        "/aside": 17,
        "audio": 18,
        "/audio": 19,
        "b": 20,
        "/b": 21,
        "base": 22,
        "/base": 23,
        "basefont": 24,
        "/basefont": 25,
        "bdi": 26,
        "/bdi": 27,
        "bdo": 28,
        "/bdo": 29,
        "big": 30,
        "/big": 31,
        "blockquote": 32,
        "/blockquote": 33,
        "body": 34,
        "/body": 35,
        "br": 36,
        "/br": 37,
        "button": 38,
        "/button": 39,
        "canvas": 40,
        "/canvas": 41,
        "caption": 42,
        "/caption": 43,
        "center": 44,
        "/center": 45,
        "cite": 46,
        "/cite": 47,
        "code": 48,
        "/code": 49,
        "col": 50,
        "/col": 51,
        "colgroup": 52,
        "/colgroup": 53,
        "data": 54,
        "/data": 55,
        "datalist": 56,
        "/datalist": 57,
        "dd": 58,
        "/dd": 59,
        "del": 60,
        "/del": 61,
        "details": 62,
        "/details": 63,
        "dfn": 64,
        "/dfn": 65,
        "dialog": 66,
        "/dialog": 67,
        "dir": 68,
        "/dir": 69,
        "div": 70,
        "/div": 71,
        "dl": 72,
        "/dl": 73,
        "dt": 74,
        "/dt": 75,
        "em": 76,
        "/em": 77,
        "embed": 78,
        "/embed": 79,
        "fieldset": 80,
        "/fieldset": 81,
        "figcaption": 82,
        "/figcaption": 83,
        "figure": 84,
        "/figure": 85,
        "font": 86,
        "/font": 87,
        "footer": 88,
        "/footer": 89,
        "form": 90,
        "/form": 91,
        "frame": 92,
        "/frame": 93,
        "frameset": 94,
        "/frameset": 95,
        "h1": 96,
        "/h1": 97,
        "h2": 98,
        "/h2": 99,
        "h3": 100,
        "/h3": 101,
        "h4": 102,
        "/h4": 103,
        "h5": 104,
        "/h5": 105,
        "h6": 106,
        "/h6": 107,
        "head": 108,
        "/head": 109,
        "header": 110,
        "/header": 111,
        "hr": 112,
        "/hr": 113,
        "html": 114,
        "/html": 115,
        "i": 116,
        "/i": 117,
        "iframe": 118,
        "/iframe": 119,
        "img": 120,
        "/img": 121,
        "input": 122,
        "/input": 123,
        "ins": 124,
        "/ins": 125,
        "kbd": 126,
        "/kbd": 127,
        "label": 128,
        "/label": 129,
        "legend": 130,
        "/legend": 131,
        "li": 132,
        "/li": 133,
        "link": 134,
        "/link": 135,
        "main": 136,
        "/main": 137,
        "map": 138,
        "/map": 139,
        "mark": 140,
        "/mark": 141,
        "meta": 142,
        "/meta": 143,
        "meter": 144,
        "/meter": 145,
        "nav": 146,
        "/nav": 147,
        "noframes": 148,
        "/noframes": 149,
        "noscript": 150,
        "/noscript": 151,
        "object": 152,
        "/object": 153,
        "ol": 154,
        "/ol": 155,
        "optgroup": 156,
        "/optgroup": 157,
        "option": 158,
        "/option": 159,
        "output": 160,
        "/output": 161,
        "p": 162,
        "/p": 163,
        "param": 164,
        "/param": 165,
        "picture": 166,
        "/picture": 167,
        "pre": 168,
        "/pre": 169,
        "progress": 170,
        "/progress": 171,
        "q": 172,
        "/q": 173,
        "rp": 174,
        "/rp": 175,
        "rt": 176,
        "/rt": 177,
        "ruby": 178,
        "/ruby": 179,
        "s": 180,
        "/s": 181,
        "samp": 182,
        "/samp": 183,
        "script": 184,
        "/script": 185,
        "section": 186,
        "/section": 187,
        "select": 188,
        "/select": 189,
        "small": 190,
        "/small": 191,
        "source": 192,
        "/source": 193,
        "span": 194,
        "/span": 195,
        "strike": 196,
        "/strike": 197,
        "strong": 198,
        "/strong": 199,
        "style": 200,
        "/style": 201,
        "sub": 202,
        "/sub": 203,
        "summary": 204,
        "/summary": 205,
        "sup": 206,
        "/sup": 207,
        "svg": 208,
        "/svg": 209,
        "table": 210,
        "/table": 211,
        "tbody": 212,
        "/tbody": 213,
        "td": 214,
        "/td": 215,
        "template": 216,
        "/template": 217,
        "textarea": 218,
        "/textarea": 219,
        "tfoot": 220,
        "/tfoot": 221,
        "th": 222,
        "/th": 223,
        "thead": 224,
        "/thead": 225,
        "time": 226,
        "/time": 227,
        "title": 228,
        "/title": 229,
        "tr": 230,
        "/tr": 231,
        "track": 232,
        "/track": 233,
        "tt": 234,
        "/tt": 235,
        "u": 236,
        "/u": 237,
        "ul": 238,
        "/ul": 239,
        "var": 240,
        "/var": 241,
        "video": 242,
        "/video": 243,
        "wbr": 244,
        "/wbr": 245
    }

    @staticmethod
    def to_char(tag):
        return chr(TagTable.TAG_MAP.get(tag, 0))


class TagSequenceState(WebState):
    """HTML tag-sequence abstraction used by WebExplor-style baselines.

    The WebExplor paper first gates state matching by URL equality and only
    then applies tag-sequence similarity. The URL gate is opt-in so older
    tag-sequence-only variants keep their old meaning.
    """

    _dump_counter = 0

    def get_action_list(self) -> List[WebAction]:
        pass

    def get_action_detailed_data(self) -> Tuple[Dict[WebAction, Any], Any]:
        pass

    def update_action_execution_time(self, action: WebAction) -> None:
        pass

    def update_transition_information(self, action: WebAction, new_state: 'WebState') -> None:
        pass

    def __lt__(self, other: object) -> bool:
        pass

    def __init__(self, html, url=None, require_same_url=None):
        fair_mode = os.environ.get("WEBTEST_FAIR_MODE", "0").strip().lower() in ("1", "true", "yes", "on")
        allow_webexplor = (
            os.environ.get("WEBTEST_ALLOW_TAG_SEQUENCE_IN_FAIR_WEBEXPLOR", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if fair_mode and not allow_webexplor:
            raise RuntimeError("TagSequenceState is legacy-only and cannot be used in fair mode")
        self.url = str(url or "").strip()
        if require_same_url is None:
            require_same_url = (
                os.environ.get("WEBTEST_TAG_SEQUENCE_REQUIRE_SAME_URL", "").strip().lower()
                in ("1", "true", "yes", "on")
            )
        self.require_same_url = bool(require_same_url)
        self.mapped_tags = to_mapped_tags(html)
        self._print_mapped_tags_if_enabled()
        self._dump_mapped_tags_if_enabled(html)
        self.sim_dic = defaultdict(float)

    def _print_mapped_tags_if_enabled(self) -> None:
        enabled = os.environ.get("WEBTEST_PRINT_TAG_SEQUENCE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not enabled:
            return

        max_len = _read_positive_int("WEBTEST_TAG_SEQUENCE_PRINT_MAX_LEN", 1000)
        mapped = self.mapped_tags
        if max_len and len(mapped) > max_len:
            mapped_for_output = mapped[:max_len]
            suffix = f"...<truncated {len(mapped) - max_len} chars>"
        else:
            mapped_for_output = mapped
            suffix = ""

        escaped = mapped_for_output.encode("unicode_escape").decode("ascii")
        print(
            "[TagSequenceState] mapped_tag_sequence_len={} mapped_tag_sequence_string={}{}".format(
                len(mapped),
                escaped,
                suffix,
            ),
            flush=True,
        )

    def _dump_mapped_tags_if_enabled(self, html) -> None:
        enabled = os.environ.get("WEBTEST_DUMP_TAG_SEQUENCE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not enabled:
            return

        max_states = _read_positive_int("WEBTEST_TAG_SEQUENCE_DUMP_MAX_STATES", 200)
        if max_states and TagSequenceState._dump_counter >= max_states:
            return

        dump_dir = os.environ.get("WEBTEST_TAG_SEQUENCE_DUMP_DIR", "").strip()
        if not dump_dir:
            dump_dir = os.path.join(os.getcwd(), "webtest_output", "tag_sequence_dumps")
        os.makedirs(dump_dir, exist_ok=True)

        TagSequenceState._dump_counter += 1
        max_len = _read_positive_int("WEBTEST_TAG_SEQUENCE_DUMP_MAX_LEN", 20000)
        mapped = self.mapped_tags
        truncated = False
        if max_len and len(mapped) > max_len:
            mapped_for_output = mapped[:max_len]
            truncated = True
        else:
            mapped_for_output = mapped

        record = {
            "index": TagSequenceState._dump_counter,
            "url": self.url,
            "require_same_url": self.require_same_url,
            "html_sha1": hashlib.sha1(str(html or "").encode("utf-8", errors="ignore")).hexdigest(),
            "mapped_sha1": hashlib.sha1(mapped.encode("latin-1", errors="ignore")).hexdigest(),
            "mapped_length": len(mapped),
            "truncated": truncated,
            "mapped_tags_ord": [ord(ch) for ch in mapped_for_output],
            "mapped_tags_hex": "".join(f"{ord(ch):02x}" for ch in mapped_for_output),
        }

        file_name = f"tag_sequence_{os.getpid()}.jsonl"
        with open(os.path.join(dump_dir, file_name), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def __eq__(self, other):
        if isinstance(other, TagSequenceState):
            if self.require_same_url or other.require_same_url:
                return self.url == other.url and self.mapped_tags == other.mapped_tags
            return self.mapped_tags == other.mapped_tags
        return False

    def __hash__(self):
        if self.require_same_url:
            return hash((self.url, self.mapped_tags))
        return hash(self.mapped_tags)

    def similarity(self, other):
        if not isinstance(other, TagSequenceState):
            return 0
        if (self.require_same_url or other.require_same_url) and self.url != other.url:
            return 0.0
        if not self.sim_dic.__contains__(other):
            sim = tag_similarity(self.mapped_tags, other.mapped_tags)
            self.sim_dic[other] = sim
            other.sim_dic[self] = sim
        return self.sim_dic[other]


def to_mapped_tags(html):
    result = []
    pattern = re.compile(r"<!(--)?(.*?)-->|<(/?\w+).*?(/?)>")
    matches = pattern.finditer(html)
    for match in matches:
        g2 = match.group(2)
        if g2 is None:  # g1 != None 表示匹配到了HTML注释
            g3 = match.group(3)
            g4 = match.group(4)
            tag = g4 + g3  # 生成 /a 而不是 a/
            result.append(TagTable.to_char(tag))  # 假设TagTable.to_char(tag)是转换标签的函数

    return ''.join(result)


def _downsample_sequence(seq: str, max_len: int = 4000) -> str:
    if len(seq) <= max_len:
        return seq
    step = max(1, len(seq) // max_len)
    return seq[::step]


def _read_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else 0


def tag_similarity(s1: str, s2: str) -> float:
    if s1 is None:
        raise ValueError("s1 must not be null")
    if s2 is None:
        raise ValueError("s2 must not be null")
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Fast, bounded-cost similarity for large HTML tag sequences.
    s1_ds = _downsample_sequence(s1)
    s2_ds = _downsample_sequence(s2)
    return difflib.SequenceMatcher(None, s1_ds, s2_ds, autojunk=False).ratio()
