"""Inspectable prompt templates with strict one-pass variable rendering."""

from __future__ import annotations

import re

from .agent import PROMPT as AGENT_PROMPT
from .editing import PROMPT as EDITING_PROMPT
from .generation import PROMPT as GENERATION_PROMPT
from .verification import PROMPT as VERIFICATION_PROMPT
from .visual_description import PROMPT as VISUAL_DESCRIPTION_PROMPT

_PLACEHOLDER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def render_prompt(template: str, /, *, prompt_name: str, **values: object) -> str:
    """Render an imported template without interpreting inserted content."""
    expected = set(_PLACEHOLDER.findall(template))
    supplied = set(values)
    if expected != supplied:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            f"prompt variables do not match {prompt_name}: missing={missing}, extra={extra}"
        )
    return _PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), template).strip()


__all__ = [
    "AGENT_PROMPT",
    "EDITING_PROMPT",
    "GENERATION_PROMPT",
    "VERIFICATION_PROMPT",
    "VISUAL_DESCRIPTION_PROMPT",
    "render_prompt",
]
