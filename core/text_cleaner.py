"""Step 2 - Text cleaning (fix formatting noise without destroying meaning)."""
from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LIST_START = re.compile(r"^\s*(?:[-*\u2022\u25cf\u25aa]|\d+[.)]|[a-zA-Z][.)])\s+")


def clean_text(text: str) -> str:
    """Normalise whitespace, repair broken lines, keep punctuation and paragraph structure."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = _CONTROL_CHARS.sub("", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines

    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):  # blank line = real paragraph break
        merged = ""
        for raw in block.split("\n"):
            line = re.sub(r"[ \t]+", " ", raw).strip()
            if not line:
                continue
            if not merged:
                merged = line
            elif _LIST_START.match(line):
                merged += "\n" + line  # keep bullets / numbered items on their own line
            else:
                merged += " " + line  # a single newline inside a paragraph is just a wrap
        if merged:
            paragraphs.append(merged)

    return "\n\n".join(paragraphs).strip()


def clean_question(question: str) -> str:
    return re.sub(r"\s+", " ", _CONTROL_CHARS.sub("", question or "")).strip()
