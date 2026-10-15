"""Limits on a deliverable that the server measures itself."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Mapping

from backend.app.generation import ArtifactBlock, DeliverableConstraint, GeneratedArtifact

from .numbers import source_values

SPOKEN_WORDS_PER_MINUTE = 150
DURATION_TOLERANCE = 0.15


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """One limit, what it requires, and what the draft measured."""

    kind: str
    scope: str
    basis: str
    quote: str
    required: str
    measured: str
    met: bool
    fix: str = ""
    enforced: bool = True

    @property
    def feedback(self) -> str:
        return " ".join(
            part for part in (
                f"Required: {self.required}.", f"The draft has {self.measured}.", self.fix,
            ) if part
        )


def word_target(seconds: int) -> int:
    """The spoken word count that takes `seconds` to say at the planning pace."""
    return round(seconds * SPOKEN_WORDS_PER_MINUTE / 60)


ENFORCED_KINDS = ("item_count", "max_characters", "max_words")

DEFAULT_WORDS_PER_ITEM = 150
CHARACTERS_PER_WORD = 6


def requested_words(constraints: Iterable[DeliverableConstraint]) -> int | None:
    """How many words the request calls for, when its limits say; None otherwise."""
    limits = tuple(constraints)
    count = next((item.value for item in limits if item.kind == "item_count"), None)
    estimates = [count * DEFAULT_WORDS_PER_ITEM] if count else []
    for item in limits:
        repeat = (count or 1) if item.scope == "each_item" else 1
        if item.kind == "duration_seconds":
            estimates.append(word_target(item.value) * repeat)
        elif item.kind == "max_words":
            estimates.append(item.value * repeat)
        elif item.kind == "max_characters":
            estimates.append(math.ceil(item.value / CHARACTERS_PER_WORD) * repeat)
    return max(estimates) if estimates else None


def word_band(seconds: int) -> tuple[int, int]:
    """The spoken word count that takes about `seconds` to say."""
    target = seconds * SPOKEN_WORDS_PER_MINUTE / 60
    low = max(1, math.ceil(target * (1 - DURATION_TOLERANCE)))
    return low, max(low, math.floor(target * (1 + DURATION_TOLERANCE)))


def _squash(text: str) -> str:
    return " ".join(text.casefold().split())


def _unanchored(constraint: DeliverableConstraint, message: str) -> str | None:
    """Why this limit cannot be trusted, or None when the user's words back it."""
    if _squash(constraint.quote) not in _squash(message):
        return "the quoted words are not in the user's message"
    if constraint.basis == "implied":
        return None
    stated = source_values([constraint.quote])
    value = Decimal(constraint.value)
    readings = {value}
    if constraint.kind == "duration_seconds":
        readings |= {value / 60, value / 3600}
    if not readings & stated:
        return "the quoted words do not state this number"
    return None


def parse_constraints(
    raw: Any, message: str
) -> tuple[tuple[DeliverableConstraint, ...], tuple[dict[str, Any], ...]]:
    """Keep the planner's limits that the user's own words back; report the rest."""
    accepted: list[DeliverableConstraint] = []
    ignored: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, (list, tuple)) else ():
        if not isinstance(item, Mapping):
            continue
        kind = item.get("kind")
        try:
            constraint = DeliverableConstraint(
                kind=kind, value=item.get("value"), quote=str(item.get("quote") or ""),
                scope="whole" if kind == "item_count" else (item.get("scope") or "whole"),
                basis=item.get("basis") or "stated",
            )
        except (TypeError, ValueError) as error:
            reason = str(error)
        else:
            reason = _unanchored(constraint, message)
            if reason is None and any(
                (kept.kind, kept.scope) == (constraint.kind, constraint.scope)
                for kept in accepted
            ):
                reason = "another limit of the same kind was already accepted"
            if reason is None:
                accepted.append(constraint)
                continue
        ignored.append({
            "kind": kind, "value": item.get("value"), "quote": item.get("quote"),
            "reason": reason,
        })
    return tuple(accepted), tuple(ignored)


def _where(constraint: DeliverableConstraint) -> str:
    return "in each item" if constraint.scope == "each_item" else "in total"


def requirement(constraint: DeliverableConstraint) -> str:
    """The limit in the exact words given to the writer and reported to the user."""
    value = constraint.value
    if constraint.kind == "item_count":
        return f"exactly {value} item{'s' if value != 1 else ''}, one block per item"
    if constraint.kind == "max_characters":
        return f"at most {value} characters {_where(constraint)}"
    if constraint.kind == "max_words":
        return f"at most {value} words {_where(constraint)}"
    low, high = word_band(value)
    return (
        f"about {word_target(value)} spoken words {_where(constraint)}, at least {low} "
        f"and at most {high} ({value} seconds at {SPOKEN_WORDS_PER_MINUTE} words per minute)"
    )


def describe_constraints(constraints: Iterable[DeliverableConstraint]) -> tuple[str, ...]:
    return tuple(
        f'{requirement(item)[0].upper()}{requirement(item)[1:]} (from "{item.quote}")'
        for item in constraints
    )


def _text(blocks: Iterable[ArtifactBlock]) -> str:
    return " ".join(unit.text for block in blocks for unit in block.units)


def _spoken_words(blocks: tuple[ArtifactBlock, ...]) -> int:
    """Words meant to be said aloud. When the writer marked none as spoken, the
    whole text is taken as the script."""
    units = [unit for block in blocks for unit in block.units]
    spoken = [unit for unit in units if (unit.delivery or "").casefold() == "spoken"]
    return sum(len(unit.text.split()) for unit in (spoken or units))


def _size(constraint: DeliverableConstraint, blocks: tuple[ArtifactBlock, ...]) -> int:
    if constraint.kind == "max_characters":
        return len(_text(blocks))
    if constraint.kind == "max_words":
        return len(_text(blocks).split())
    return _spoken_words(blocks)


def _unit_name(constraint: DeliverableConstraint, size: int) -> str:
    if constraint.kind == "max_characters":
        return f"{size} characters"
    if constraint.kind == "max_words":
        return f"{size} words"
    seconds = round(size / SPOKEN_WORDS_PER_MINUTE * 60)
    return f"{size} spoken words (about {seconds} seconds)"


def _within(constraint: DeliverableConstraint, size: int) -> bool:
    if constraint.kind == "duration_seconds":
        low, high = word_band(constraint.value)
        return low <= size <= high
    return size <= constraint.value


def _adjustment(constraint: DeliverableConstraint, size: int) -> str:
    """How much to change a size that is outside its limit, or "" when inside."""
    if _within(constraint, size):
        return ""
    if constraint.kind == "duration_seconds":
        target = word_target(constraint.value)
        if size < target:
            return f"add about {target - size} spoken words"
        return f"cut about {size - target} spoken words"
    unit = "characters" if constraint.kind == "max_characters" else "words"
    return f"cut about {size - constraint.value} {unit}"


def measure_constraints(artifact: GeneratedArtifact) -> tuple[ConstraintResult, ...]:
    """Measure the draft against every limit attached to its request."""
    results = []
    for constraint in artifact.request.output.constraints:
        if constraint.kind == "item_count":
            count = len(artifact.blocks)
            met = count == constraint.value
            measured = f"{count} item{'s' if count != 1 else ''}"
            difference = constraint.value - count
            fix = "" if met else (
                f"{'Add' if difference > 0 else 'Remove'} {abs(difference)} "
                f"item{'s' if abs(difference) != 1 else ''}: return exactly "
                f"{constraint.value} blocks."
            )
        elif constraint.scope == "each_item":
            sizes = [_size(constraint, (block,)) for block in artifact.blocks]
            outside = [
                (index, size) for index, size in enumerate(sizes, start=1)
                if not _within(constraint, size)
            ]
            met = not outside
            measured = ", ".join(
                f"item {index} with {_unit_name(constraint, size)}" for index, size in outside
            ) if outside else (
                f"every item within the limit (largest: {_unit_name(constraint, max(sizes))})"
            )
            fix = "" if met else "In " + "; in ".join(
                f"item {index}, {_adjustment(constraint, size)}" for index, size in outside
            ) + "."
        else:
            size = _size(constraint, artifact.blocks)
            met = _within(constraint, size)
            measured = _unit_name(constraint, size)
            adjustment = _adjustment(constraint, size)
            fix = f"{adjustment[0].upper()}{adjustment[1:]} in total." if adjustment else ""
        if fix and constraint.kind in {"duration_seconds", "max_words", "max_characters"}:
            fix += (
                " Spread the change through the text, and add only explanation of what "
                "the source says, never new facts."
                if "add about" in fix.casefold() else " Keep every point the request needs."
            )
        results.append(ConstraintResult(
            constraint.kind, constraint.scope, constraint.basis, constraint.quote,
            requirement(constraint), measured, met, fix,
            enforced=constraint.basis == "stated" and constraint.kind in ENFORCED_KINDS,
        ))
    return tuple(results)
