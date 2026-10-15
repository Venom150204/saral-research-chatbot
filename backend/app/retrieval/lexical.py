"""Deterministic BM25 keyword search over one authorized document version."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

from py_rust_stemmers import SnowballStemmer

from backend.app.chunking import ChunkSet
from backend.app.normalization import to_written_form

from .models import SearchHit
from .scope import validate_scope

TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)?|\w+", re.UNICODE)
_STEMMER = SnowballStemmer("english")

Tokenizer = Callable[[str], tuple[str, ...]]


def tokenize(text: str) -> tuple[str, ...]:
    """Keyword tokens: written form, case-folded, Snowball-stemmed."""
    return tuple(
        _STEMMER.stem_word(match.group(0).casefold())
        for match in TOKEN_PATTERN.finditer(to_written_form(text))
    )


class LexicalIndex:
    def __init__(
        self, chunk_set: ChunkSet, *, k1: float = 1.5, b: float = 0.75,
        tokenizer: Tokenizer = tokenize,
    ) -> None:
        if not chunk_set.chunks:
            raise ValueError("cannot index an empty chunk set")
        self._chunk_set = chunk_set
        self._k1 = k1
        self._b = b
        self._tokenizer = tokenizer
        self._tokens = [tokenizer(chunk.search_text) for chunk in chunk_set.chunks]
        self._counts = [Counter(tokens) for tokens in self._tokens]
        self._average_length = sum(map(len, self._tokens)) / len(self._tokens)
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        count = len(self._tokens)
        self._idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        document_id: str,
        document_version: str,
        top_k: int = 5,
        slide_number: int | None = None,
    ) -> tuple[SearchHit, ...]:
        validate_scope(
            self._chunk_set,
            document_id=document_id,
            document_version=document_version,
        )
        if top_k < 1:
            raise ValueError("top_k must be positive")
        terms = self._tokenizer(query)
        scored = []
        for index, (chunk, counts, tokens) in enumerate(
            zip(self._chunk_set.chunks, self._counts, self._tokens, strict=True)
        ):
            if slide_number is not None and not any(
                ref.slide_number == slide_number for ref in chunk.evidence
            ):
                continue
            score = sum(
                self._term_score(term, counts[term], len(tokens))
                for term in terms
                if counts[term]
            )
            if score > 0:
                scored.append((score, index, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(
            SearchHit(rank=rank, score=score, chunk=chunk)
            for rank, (score, _, chunk) in enumerate(scored[:top_k], start=1)
        )

    def _term_score(self, term: str, frequency: int, length: int) -> float:
        denominator = frequency + self._k1 * (
            1 - self._b + self._b * length / self._average_length
        )
        return self._idf.get(term, 0.0) * frequency * (self._k1 + 1) / denominator
