from icu import BreakIterator, Locale
import unicodedata, hashlib

def _utf16_unit_count(ch: str) -> int:
    # Number of UTF-16 code units for a single Unicode character (1 or 2)
    return len(ch.encode("utf-16-le")) // 2

def _build_utf16_to_py_map(s: str):
    """
    Map UTF-16 code-unit offsets -> Python code-point indices.
    map16[k] = Python index at the boundary of UTF-16 offset k.
    """
    total_cu = len(s.encode("utf-16-le")) // 2
    map16 = [0] * (total_cu + 1)
    cu_pos = 0
    for i, ch in enumerate(s):
        cu = _utf16_unit_count(ch)
        for _ in range(cu):
            map16[cu_pos] = i
            cu_pos += 1
    map16[cu_pos] = len(s)
    return map16

def icu_word_spans(text: str, locale_str: str = "und", keep_separators: bool = False):
    """
    Return a list of (token, start_py, end_py, is_word) using ICU word boundaries.
    start/end are Python indices (code points), not UTF-16 units.
    """
    s = unicodedata.normalize("NFC", text)
    map16 = _build_utf16_to_py_map(s)

    loc = Locale.forLanguageTag(locale_str)
    bi = BreakIterator.createWordInstance(loc)
    bi.setText(s)

    spans = []
    start16 = bi.first()
    for end16 in bi:
        start_py = map16[start16]
        end_py = map16[end16]
        token = s[start_py:end_py]
        status = bi.getRuleStatus()  # 0 = non-word, >0 = word categories
        is_word = (status != 0)
        if is_word or keep_separators:
            spans.append((token, start_py, end_py, is_word))
        start16 = end16
    return spans



def normalize_token(s: str) -> str:
    # Language-agnostic, Unicode-safe normalization
    return unicodedata.normalize("NFKC", s).casefold().strip()

def surface_id(word: str, bits: int = 64, salt: bytes = b"v1") -> int:
    """
    Deterministic ID for the exact (normalized) surface form.
    No collisions in practice at 64+ bits (but theoretically possible).
    """
    w = normalize_token(word)
    h = hashlib.blake2b(w.encode("utf-8"), digest_size=bits // 8, key=salt).digest()
    return int.from_bytes(h, "big")