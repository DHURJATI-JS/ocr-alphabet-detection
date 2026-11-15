"""Turn raw OCR output into clean A-Z text.

OCR engines without a character whitelist (Apple Vision, Windows OCR) sometimes return
look-alike glyphs: Cyrillic 'А' instead of Latin 'A', fullwidth letters, or digits like
'0' for 'O'. Everything is mapped back to plain ASCII letters here.
"""

from __future__ import annotations

import re
import string
import unicodedata

ASCII_LETTERS = frozenset(string.ascii_letters)

# Cyrillic and Greek glyphs that are visually identical to Latin letters.
_HOMOGLYPHS = {
    # Cyrillic upper
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J", "К": "K",
    "М": "M", "О": "O", "Р": "P", "Ѕ": "S", "Т": "T", "Х": "X", "У": "Y", "Ү": "Y",
    # Cyrillic lower
    "а": "a", "с": "c", "е": "e", "һ": "h", "і": "i", "ј": "j", "о": "o", "р": "p",
    "ѕ": "s", "х": "x", "у": "y", "ԁ": "d", "ԛ": "q", "ԝ": "w",
    # Greek upper
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Greek lower
    "ο": "o", "ν": "v", "κ": "k", "ι": "i", "α": "a",
}

# Non-letters that OCR commonly reports for a single letter.
_LOOKALIKES = {"0": "O", "1": "I", "|": "I", "!": "I", "5": "S", "$": "S", "8": "B", "€": "E"}

_SPACES = re.compile(r"\s+")


def normalize_char(ch: str, lookalikes: bool = False) -> str:
    """Map one character to an ASCII letter, a space, or '' (dropped).

    lookalikes=True also maps digits/punctuation that resemble a letter ('0' -> 'O'); only
    sensible for a lone glyph, not inside words ("world!" must not become "WORLDI").
    """
    ch = unicodedata.normalize("NFKC", ch)
    if len(ch) != 1:
        return "".join(normalize_char(c, lookalikes) for c in ch)
    if ch in ASCII_LETTERS:
        return ch
    if ch in _HOMOGLYPHS:
        return _HOMOGLYPHS[ch]
    if lookalikes and ch in _LOOKALIKES:
        return _LOOKALIKES[ch]
    if ch.isspace():
        return " "
    # Accented Latin letters (É -> E) after decomposition.
    base = unicodedata.normalize("NFD", ch)[0]
    return base if base in ASCII_LETTERS else ""


def apply_case(text: str, case: str) -> str:
    if case == "upper":
        return text.upper()
    if case == "lower":
        return text.lower()
    return text


def normalize_text(raw: str, case: str = "auto", lookalikes: bool = False) -> str:
    """Keep only letters (and single spaces between words), then apply the case rule."""
    text = "".join(normalize_char(ch, lookalikes) for ch in raw)
    text = _SPACES.sub(" ", text).strip()
    return apply_case(text, case)


def normalize_letter(raw: str, case: str = "auto") -> str:
    """Return exactly one letter, or '' when the reading is not a single letter."""
    letters = normalize_text(raw, "auto", lookalikes=True).replace(" ", "")
    if len(letters) != 1:
        return ""
    # A bare vertical bar is both lowercase 'l' and uppercase 'I' in sans-serif fonts, and
    # OCR usually calls it 'l'. Uppercased that would become 'L', which has a foot - wrong.
    if case == "upper" and letters == "l":
        return "I"
    if case == "lower" and letters == "I":
        return "l"
    return apply_case(letters, case)
