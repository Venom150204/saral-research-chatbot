"""Bring speech-oriented deck text to written form for matching."""

from __future__ import annotations

import re
import unicodedata

from text_to_num import alpha2digit

_SPELLED_INITIALISM = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z] )+[A-Z](?![A-Za-z0-9])")
_SPOKEN_DIGIT_DECIMAL = re.compile(r"(?<![\w.])(\d+) point ((?:\d )*\d)(?![\w.])")
_NUMERAL_WITH_SCALE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?) (thousand|million|billion|trillion)\b", re.IGNORECASE
)
_SCALES = {"thousand": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}
_SPOKEN_DECIMAL_THEN_SCALE = re.compile(
    r"\b(point(?: (?:zero|one|two|three|four|five|six|seven|eight|nine))+) "
    r"(thousand|million|billion|trillion)\b",
    re.IGNORECASE,
)
_HELD_SCALE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?) ⟦E(\d+)⟧")
_UNUSED_HELD_SCALE = re.compile(r"⟦E(\d+)⟧")
_SCALE_WORDS = {len(str(value)) - 1: word for word, value in _SCALES.items()}
_LETTER_THEN_NUMBER = re.compile(r"(?<![A-Za-z0-9])([B-HJ-Z]) (\d+)(?![\w.])")
_DIGIT_WORDS = {
    word: str(value) for value, word in enumerate(
        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    )
}
_LETTER_THEN_DIGIT_WORD = re.compile(
    r"(?<![A-Za-z0-9])([B-HJ-Z]) (" + "|".join(_DIGIT_WORDS) + r")\b"
)
_PERCENT_WORD = re.compile(r"(?<=\d) ?(?:percent|per cent)\b", re.IGNORECASE)
_VAGUE_MAGNITUDES = ("hundreds", "thousands", "millions", "billions", "trillions", "dozens", "tens")
_VAGUE_MAGNITUDE = re.compile(r"\b(" + "|".join(_VAGUE_MAGNITUDES) + r")\b", re.IGNORECASE)
_HELD_VAGUE_MAGNITUDE = re.compile(r"⟦V(\d)(\d)⟧")


def _hold_vague_magnitude(match: re.Match[str]) -> str:
    word = match.group(1)
    capitalised = int(word[0].isupper())
    return f"⟦V{_VAGUE_MAGNITUDES.index(word.lower())}{capitalised}⟧"


def _release_vague_magnitude(match: re.Match[str]) -> str:
    word = _VAGUE_MAGNITUDES[int(match.group(1))]
    return word.capitalize() if match.group(2) == "1" else word


def _plain_number(value: float) -> str:
    rounded = round(value, 6)
    return str(int(rounded)) if rounded == int(rounded) else repr(rounded)


def _scaled(match: re.Match[str]) -> str:
    return _plain_number(float(match.group(1)) * _SCALES[match.group(2).lower()])


def _scaled_by_power(match: re.Match[str]) -> str:
    return _plain_number(float(match.group(1)) * 10 ** int(match.group(2)))


def to_written_form(
    text: str,
    *,
    convert_isolated_small_numbers: bool = False,
    convert_number_words: bool = True,
) -> str:
    """Return `text` with spoken numbers and spelled initialisms in written form."""
    written = unicodedata.normalize("NFKC", text)
    written = _SPELLED_INITIALISM.sub(lambda match: match.group(0).replace(" ", ""), written)
    written = _NUMERAL_WITH_SCALE.sub(_scaled, written)
    if convert_number_words:
        written = _VAGUE_MAGNITUDE.sub(_hold_vague_magnitude, written)
        written = _SPOKEN_DECIMAL_THEN_SCALE.sub(
            lambda match: (
                f"{match.group(1)} "
                f"⟦E{len(str(_SCALES[match.group(2).lower()])) - 1}⟧"
            ),
            written,
        )
        written = alpha2digit(
            written, "en", threshold=0 if convert_isolated_small_numbers else 3
        )
        written = _HELD_SCALE.sub(_scaled_by_power, written)
        written = _UNUSED_HELD_SCALE.sub(
            lambda match: _SCALE_WORDS[int(match.group(1))], written
        )
        written = _HELD_VAGUE_MAGNITUDE.sub(_release_vague_magnitude, written)
    written = _SPOKEN_DIGIT_DECIMAL.sub(
        lambda match: f"{match.group(1)}.{match.group(2).replace(' ', '')}", written
    )
    written = _LETTER_THEN_DIGIT_WORD.sub(
        lambda match: match.group(1) + _DIGIT_WORDS[match.group(2)], written
    )
    written = _LETTER_THEN_NUMBER.sub(r"\1\2", written)
    return _PERCENT_WORD.sub("%", written)
