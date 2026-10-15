# Input format strategy

SARAL accepts decks (PPTX, a deck exported as PDF, Beamer LaTeX), papers and briefs (PDF, LaTeX article, Markdown, text).

## Support tiers

| Tier | Formats | Handling | Acceptance boundary |
|---|---|---|---|
| Native now | `.pptx` | `python-pptx` reads slide order, shape positions, paragraphs, hierarchy levels, tables, speaker notes, and visual placeholders | Stable slide/shape evidence IDs; no silent claim that image/chart meaning was read |
| Native now | `.txt`, `.md`, `.markdown` | UTF-8 paragraph/heading extraction | Stable line spans and source text |
| Native now | `.pdf` | `pypdf` text layer. Landscape pages: one slide per page (title + bullets). Portrait pages: paragraphs under detected headings. Repeated headers/footers and the reference list skipped and reported | Slide number or page number on every unit; scanned PDFs (no text layer) rejected, no OCR |
| Native now | `.tex` | Read as text, never executed. Beamer: frames as slides, `\frametitle`, `\item`, `
ote`. Article: sections and paragraphs. Display math kept verbatim as one unit | Source line range on every unit; equations never split |
| Native next | `.odp` | Dedicated OpenDocument adapter after fixtures and library spike | Slide order, text, notes, tables, and visual placeholders map to the same domain model |
| Native next | `.docx` summaries | Dedicated WordprocessingML adapter | Heading/paragraph/table provenance maps to the summary model |
| Conversion compatibility | legacy `.ppt` | Sandboxed LibreOffice conversion to `.pptx`, retaining original and conversion diagnostics | Layout/text/notes comparison passes representative fixtures; time/resource limits enforced |
| Export compatibility | Google Slides | User exports `.pptx` or `.odp`; authenticated Drive import is a later integration | Imported export hash and format are recorded |
| Export compatibility | Apple Keynote `.key` | User exports `.pptx`; native server parsing is not promised | Export limitation is explicit in UI |

“Common format support” means a tested native or conversion path, not merely accepting an extension. Native adapters are preferred when they preserve provenance. Conversion is appropriate for closed or legacy formats only after fidelity, deployment size, cold start, security, and licensing are measured.

## Deck extraction contract

- Each slide is a parent evidence unit with its original one-based slide number.
- Text blocks retain stable shape ID, z-order traversal, paragraph index, hierarchy level, and EMU bounding box.
- Table cells retain row and column coordinates; headers are not guessed during extraction.
- Speaker notes remain a distinct block type.
- Pictures and charts become visual placeholders with shape identity and location. Embedded pictures also retain content hash, MIME type, byte size, pixel dimensions, and an internal asset URI when storage is configured. Native chart titles, categories, series, and values are serialized into Markdown when accessible. Visual semantics remain model-derived rather than original text.
- Grouped shapes are traversed without flattening slide identity.
- Unsupported equations, SmartArt, embedded objects, chart series, animations, and alt-text gaps are recorded as extraction limitations before retrieval.

## Summary extraction contract

- Plain text and Markdown are decoded strictly as UTF-8.
- Paragraphs/headings retain one-based source line spans.
- A summary is user-provided evidence; it supports what the summary says, not necessarily the underlying research.
- DOCX support will preserve paragraphs, headings, tables, and their document order rather than flattening them blindly.

## Fixture matrix

The initial deck fixtures must include:

1. title, nested bullets, and ordinary text;
2. numerical table with headers and units;
3. speaker notes containing a qualification absent from the visible slide;
4. picture-only slide and chart-heavy slide;
5. grouped and overlapping shapes with non-reading-order insertion;
6. equation/embedded-object slide to expose unsupported content;
7. empty, malformed, oversized, and excessive-slide uploads;
8. repeated upload proving stable IDs.

Summary fixtures include structured Markdown, long plain text, Unicode, invalid encoding, empty input, and adversarial “ignore instructions” content that remains data.
