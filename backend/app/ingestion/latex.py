"""LaTeX extraction: a Beamer deck (one slide per frame) or an article (sections)."""

from __future__ import annotations

from bisect import bisect_right
from hashlib import sha256
from pathlib import Path
import re

from backend.app.domain import ExtractedDocument, ExtractionWarning, SourceBlock, SourceLocation

from .errors import ExtractionQualityError, UnsupportedFormatError

MAX_LATEX_BYTES = 2 * 1024 * 1024
MATH_ENVIRONMENTS = {
    "equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*",
    "eqnarray", "eqnarray*", "displaymath", "flalign", "flalign*",
}
FLOAT_ENVIRONMENTS = {"figure", "figure*", "table", "table*"}
HEADINGS = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}
TOKEN = re.compile(
    r"\\(?P<env_cmd>begin|end)\{(?P<env>[^}]+)\}"
    r"|\\(?P<cmd>section|subsection|subsubsection|paragraph|frametitle|caption|note|item|bibliography)\*?"
    r"|(?P<display>\\\[|\$\$)"
)
COMMENT = re.compile(r"(?<!\\)%.*$")
BEAMER = re.compile(r"\\documentclass(?:\[[^\]]*\])?\{beamer\}|\\begin\{frame\}")


def extract_latex_bytes(payload: bytes, *, filename: str) -> ExtractedDocument:
    safe_filename = Path(filename).name
    if not payload:
        raise ExtractionQualityError("LaTeX source is empty")
    if len(payload) > MAX_LATEX_BYTES:
        raise UnsupportedFormatError(f"LaTeX source is {len(payload)} bytes; limit is {MAX_LATEX_BYTES} bytes")
    try:
        source = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        source = payload.decode("latin-1")
    text = "\n".join(COMMENT.sub("", line) for line in source.splitlines())
    content_hash = sha256(payload).hexdigest()
    parser = _Parser(text, content_hash, beamer=bool(BEAMER.search(text)))
    parser.run()
    if not parser.blocks:
        raise ExtractionQualityError("no readable text was found in the LaTeX source")
    deck = parser.beamer
    return ExtractedDocument(
        document_id=f"doc_{content_hash[:16]}",
        document_version=f"sha256:{content_hash}",
        content_hash=content_hash,
        filename=safe_filename,
        media_type="application/x-tex",
        source_kind="slide_deck" if deck else "document",
        parser_name="saral-latex",
        parser_version="1",
        unit_count=max(1, parser.slide if deck else parser.sections),
        blocks=tuple(parser.blocks),
        warnings=tuple(parser.warnings),
    )


def braced(text: str, start: int) -> tuple[str, int] | None:
    """The balanced {...} argument beginning at `start` (after optional spaces and
    one [...] option), and the index just after it."""
    index = start
    while index < len(text) and text[index] in " \t\n":
        index += 1
    if index < len(text) and text[index] == "[":
        close = text.find("]", index)
        if close == -1:
            return None
        index = close + 1
        while index < len(text) and text[index] in " \t\n":
            index += 1
    if index >= len(text) or text[index] != "{":
        return None
    depth = 0
    for position in range(index, len(text)):
        character = text[position]
        if character == "{" and text[position - 1] != "\\":
            depth += 1
        elif character == "}" and text[position - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[index + 1:position], position + 1
    return None


_FORMATTING = re.compile(r"\\(?:textbf|textit|emph|texttt|textsc|underline|mathrm|text|textrm|alert|structure)\{([^{}]*)\}")
_DROP_WITH_ARGUMENT = re.compile(r"\\(?:cite[tp]?|citeauthor|label|vspace|hspace|includegraphics|footnote)\*?(?:\[[^\]]*\])?\{[^{}]*\}")
_REFERENCE = re.compile(r"\\(?:eqref|ref|autoref|cref)\{[^{}]*\}")
_HREF = re.compile(r"\\href\{[^{}]*\}\{([^{}]*)\}")
_URL = re.compile(r"\\url\{([^{}]*)\}")
_BARE_COMMAND = re.compile(r"\\(?:centering|small|footnotesize|large|Large|pause|noindent|maketitle|titlepage|tableofcontents|medskip|bigskip|smallskip|newpage|clearpage|hline|par)\b")


def clean_text(text: str) -> str:
    """Readable prose from LaTeX markup, keeping inline math ($...$) exactly."""
    parts = re.split(r"(\$[^$]+\$)", text)
    cleaned = []
    for index, part in enumerate(parts):
        if index % 2:
            cleaned.append(part)
            continue
        for _ in range(3):
            part = _FORMATTING.sub(r"\1", part)
        part = _DROP_WITH_ARGUMENT.sub("", part)
        part = _REFERENCE.sub("(ref.)", part)
        part = _HREF.sub(r"\1", part)
        part = _URL.sub(r"\1", part)
        part = _BARE_COMMAND.sub("", part)
        part = part.replace("~", " ").replace("\\\\", " ").replace("---", "—").replace("--", "–")
        part = re.sub(r"\\([%&#_$])", r"\1", part)
        part = part.replace("``", "\u201c").replace("''", "\u201d")
        cleaned.append(part)
    return " ".join("".join(cleaned).split())


class _Parser:
    def __init__(self, text: str, content_hash: str, *, beamer: bool) -> None:
        self.text = text
        self.hash = content_hash
        self.beamer = beamer
        self.blocks: list[SourceBlock] = []
        self.warnings: list[ExtractionWarning] = []
        self.line_starts = [0] + [match.end() for match in re.finditer("\n", text)]
        self.slide = 0
        self.sections = 0
        self.parent = f"tex_{content_hash[:12]}_0"
        self.in_frame = False

    def line(self, offset: int) -> int:
        return bisect_right(self.line_starts, offset)

    def add(self, text: str, block_type: str, start: int, end: int, level: int | None = None) -> None:
        text = text.strip()
        if not text:
            return
        location = SourceLocation(
            slide_number=self.slide if self.beamer and self.slide else None,
            line_start=self.line(start), line_end=max(self.line(start), self.line(max(start, end - 1))),
        )
        identity = f"{self.hash}\0{len(self.blocks)}\0{block_type}\0{text}".encode("utf-8")
        self.blocks.append(SourceBlock(
            block_id=f"blk_{sha256(identity).hexdigest()[:20]}", parent_id=self.parent,
            ordinal=len(self.blocks), text=text, location=location, block_type=block_type,
            hierarchy_level=level,
        ))

    def prose(self, start: int, end: int) -> None:
        """Paragraphs (split at blank lines) of the text between two structures."""
        if self.beamer and not self.in_frame:
            return
        chunk = self.text[start:end]
        offset = start
        for piece in re.split(r"(\n[ \t]*\n)", chunk):
            if piece.strip() and not re.fullmatch(r"\n[ \t]*\n", piece):
                cleaned = clean_text(piece)
                if cleaned:
                    start = offset + len(piece) - len(piece.lstrip())
                    end = offset + len(piece.rstrip())
                    self.add(cleaned, "slide_text" if self.beamer else "paragraph", start, end,
                             0 if self.beamer else None)
            offset += len(piece)

    def run(self) -> None:
        text = self.text
        title = re.search(r"\\title(?:\[[^\]]*\])?(?=\{)", text)
        begin = re.search(r"\\begin\{document\}", text)
        position = begin.end() if begin else 0
        if title and (not begin or title.start() < begin.start()):
            argument = braced(text, title.end())
            if argument:
                self.title = (clean_text(argument[0]), title.start(), argument[1])
        else:
            self.title = None
        if self.title and not self.beamer:
            self.add(self.title[0], "title", self.title[1], self.title[2])
        while True:
            match = TOKEN.search(text, position)
            if match is None:
                self.prose(position, len(text))
                return
            self.prose(position, match.start())
            position = self.handle(match)
            if position < 0:
                return

    def handle(self, match: re.Match) -> int:
        text = self.text
        environment, command, display = match.group("env"), match.group("cmd"), match.group("display")
        if display:
            closer = "\\]" if display == "\\[" else "$$"
            end = text.find(closer, match.end())
            end = len(text) if end == -1 else end + len(closer)
            if not self.beamer or self.in_frame:
                self.add(text[match.start():end], "equation", match.start(), end)
            return end
        if environment:
            name = environment.strip()
            if match.group("env_cmd") == "begin":
                if name in MATH_ENVIRONMENTS:
                    closing = text.find(f"\\end{{{name}}}", match.end())
                    end = len(text) if closing == -1 else closing + len(f"\\end{{{name}}}")
                    if not self.beamer or self.in_frame:
                        self.add(text[match.start():end], "equation", match.start(), end)
                    return end
                if name in FLOAT_ENVIRONMENTS:
                    closing = text.find(f"\\end{{{name}}}", match.end())
                    end = len(text) if closing == -1 else closing + len(f"\\end{{{name}}}")
                    caption = re.search(r"\\caption(?:\[[^\]]*\])?(?=\{)", text[match.end():end])
                    if caption:
                        argument = braced(text, match.end() + caption.end())
                        if argument:
                            self.add(clean_text(argument[0]), "caption", match.end() + caption.start(), argument[1])
                    return end
                if name == "thebibliography":
                    self.warnings.append(ExtractionWarning("references_skipped", "The bibliography is not indexed."))
                    return -1
                if name == "abstract" and not self.beamer:
                    self.heading("Abstract", 1, match.start(), match.end())
                if name == "frame":
                    self.slide += 1
                    self.in_frame = True
                    self.parent = f"tex_{self.hash[:12]}_slide{self.slide}"
                    argument = braced(text, match.end())
                    if argument:
                        self.add(clean_text(argument[0]), "title", match.end(), argument[1])
                        return argument[1]
                    if self.slide == 1 and self.title and "\\titlepage" in text[match.end():match.end() + 200]:
                        self.add(self.title[0], "title", self.title[1], self.title[2])
                return match.end()
            if name == "frame":
                self.in_frame = False
            if name == "document":
                return -1
            return match.end()
        argument = braced(text, match.end())
        end = argument[1] if argument else match.end()
        if command in HEADINGS and argument:
            self.heading(clean_text(argument[0]), HEADINGS[command], match.start(), end)
        elif command == "frametitle" and argument:
            self.add(clean_text(argument[0]), "title", match.start(), end)
        elif command == "note" and argument:
            if self.beamer:
                self.add(clean_text(argument[0]), "speaker_note", match.start(), end)
        elif command == "caption" and argument:
            self.add(clean_text(argument[0]), "caption", match.start(), end)
        elif command == "bibliography":
            return end
        elif command == "item":
            label = re.match(r"\s*\[([^\]]*)\]", text[match.end():])
            return match.end() + (label.end() if label else 0)
        return end

    def heading(self, title: str, level: int, start: int, end: int) -> None:
        if self.beamer:
            return
        self.sections += 1
        self.parent = f"tex_{self.hash[:12]}_{self.sections}"
        self.add(title, "heading", start, end, level)
