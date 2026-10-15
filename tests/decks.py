"""Small decks and pictures built in code for tests (not collected as tests)."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Any
from pptx import Presentation
from pptx.chart.data import ChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches


@dataclass(frozen=True, slots=True)
class ExpectedFact:
    fact_id: str
    alternatives: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class VisualFixture:
    fixture_id: str
    image_bytes: bytes
    slide_context: str
    expected_facts: tuple[ExpectedFact, ...]
    expected_labels: tuple[str, ...]

def build_visual_fixtures() -> tuple[VisualFixture, ...]:
    return (
        _architecture_flow(),
        _scientific_panel(),
        _raster_chart(),
        _ui_screenshot(),
        _equation_image(),
        _decorative_scene(),
        _visual_only_slide(),
    )

def _canvas(width: int = 1000, height: int = 600) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)

def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)

def _png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()

def _architecture_flow() -> VisualFixture:
    image, draw = _canvas(900, 300)
    boxes = [(40, 100, 230, 200), (355, 100, 545, 200), (670, 100, 860, 200)]
    labels = ("UPLOAD", "PARSE", "RETRIEVE")
    for box, label in zip(boxes, labels, strict=True):
        draw.rounded_rectangle(box, radius=15, outline="navy", width=5, fill="#e8f1ff")
        draw.text((box[0] + 42, box[1] + 34), label, fill="black", font=_font(24))
    for start, end in ((230, 355), (545, 670)):
        draw.line((start, 150, end, 150), fill="navy", width=6)
        draw.polygon([(end, 150), (end - 25, 136), (end - 25, 164)], fill="navy")
    return VisualFixture(
        "architecture_flow",
        _png(image),
        "SARAL document-processing architecture.",
        (
            ExpectedFact("upload_to_parse", ("upload box points to parse", "upload to parse")),
            ExpectedFact("parse_to_retrieve", ("parse box points to retrieve", "parse to retrieve")),
        ),
        labels,
    )

def _scientific_panel() -> VisualFixture:
    image, draw = _canvas()
    draw.text((30, 20), "CELL RESPONSE STUDY", fill="black", font=_font(30))
    draw.rectangle((40, 90, 470, 540), outline="black", width=3)
    draw.text((55, 105), "A", fill="black", font=_font(28))
    draw.text((160, 105), "CONTROL", fill="black", font=_font(22))
    for x, y in ((130, 220), (260, 300), (360, 190), (190, 410)):
        draw.ellipse((x - 35, y - 35, x + 35, y + 35), fill="#7fb3d5", outline="navy")
    draw.rectangle((530, 90, 960, 540), outline="black", width=3)
    draw.text((545, 105), "B", fill="black", font=_font(28))
    draw.text((650, 105), "TREATED", fill="black", font=_font(22))
    for x, y in ((620, 200), (730, 250), (850, 205), (650, 390), (790, 410), (890, 330)):
        draw.ellipse((x - 22, y - 22, x + 22, y + 22), fill="#e67e22", outline="#943126")
    draw.line((700, 500, 880, 500), fill="black", width=4)
    draw.text((745, 510), "50 um", fill="black", font=_font(18))
    return VisualFixture(
        "scientific_panel",
        _png(image),
        "Synthetic microscopy-style comparison; describe only visible differences.",
        (
            ExpectedFact("panel_a_control", ("panel a", "a control")),
            ExpectedFact("panel_b_treated", ("panel b", "b treated")),
            ExpectedFact("more_orange_in_treated", ("more orange", "six orange", "treated panel contains")),
            ExpectedFact("scale_bar", ("50 um", "50 μm", "scale bar")),
        ),
        ("A", "CONTROL", "B", "TREATED", "50 um"),
    )

def _raster_chart() -> VisualFixture:
    image, draw = _canvas()
    draw.text((220, 25), "RETRIEVAL ACCURACY BY METHOD", fill="black", font=_font(30))
    draw.line((110, 90, 110, 510), fill="black", width=4)
    draw.line((110, 510, 920, 510), fill="black", width=4)
    draw.text((20, 270), "ACCURACY %", fill="black", font=_font(20))
    values = (("LEXICAL", 62, "#7f8c8d"), ("VECTOR", 78, "#3498db"), ("HYBRID", 91, "#27ae60"))
    for index, (label, value, color) in enumerate(values):
        left = 190 + index * 245
        top = 510 - value * 4
        draw.rectangle((left, top, left + 120, 510), fill=color, outline="black")
        draw.text((left + 36, top - 32), str(value), fill="black", font=_font(22))
        draw.text((left, 530), label, fill="black", font=_font(20))
    return VisualFixture(
        "raster_chart",
        _png(image),
        "Rasterized evaluation chart.",
        (
            ExpectedFact("lexical_62", ("lexical 62", "lexical: 62", "62%")),
            ExpectedFact("vector_78", ("vector 78", "vector: 78", "78%")),
            ExpectedFact("hybrid_91", ("hybrid 91", "hybrid: 91", "91%")),
            ExpectedFact("hybrid_highest", ("hybrid is highest", "hybrid has the highest", "highest accuracy")),
        ),
        ("RETRIEVAL ACCURACY BY METHOD", "ACCURACY %", "LEXICAL", "VECTOR", "HYBRID", "62", "78", "91"),
    )

def _ui_screenshot() -> VisualFixture:
    image, draw = _canvas()
    draw.rectangle((0, 0, 1000, 70), fill="#17202a")
    draw.text((30, 20), "SARAL WORKSPACE", fill="white", font=_font(26))
    draw.text((40, 110), "Upload slide deck", fill="black", font=_font(28))
    draw.rectangle((40, 170, 720, 235), outline="#566573", width=3)
    draw.text((60, 188), "research_review.pptx", fill="#273746", font=_font(20))
    draw.rectangle((750, 170, 950, 235), fill="#2874a6")
    draw.text((805, 188), "UPLOAD", fill="white", font=_font(20))
    draw.rectangle((40, 285, 950, 500), fill="#eafaf1", outline="#239b56", width=3)
    draw.text((65, 315), "STATUS: READY", fill="#196f3d", font=_font(24))
    draw.text((65, 370), "12 slides parsed", fill="black", font=_font(20))
    draw.text((65, 410), "3 visuals need review", fill="black", font=_font(20))
    return VisualFixture(
        "ui_screenshot",
        _png(image),
        "Synthetic upload interface screenshot.",
        (
            ExpectedFact("file", ("research_review.pptx",)),
            ExpectedFact("ready", ("status: ready", "status ready")),
            ExpectedFact("slides", ("12 slides",)),
            ExpectedFact("visual_review", ("3 visuals", "three visuals")),
        ),
        ("SARAL WORKSPACE", "Upload slide deck", "research_review.pptx", "UPLOAD", "STATUS: READY", "12 slides parsed", "3 visuals need review"),
    )

def _equation_image() -> VisualFixture:
    image, draw = _canvas()
    draw.text((250, 90), "EQUATION OF MOTION", fill="black", font=_font(30))
    draw.text((250, 245), "s = u t + 1/2 a t^2", fill="navy", font=_font(48))
    draw.text((250, 350), "where a = acceleration", fill="black", font=_font(24))
    return VisualFixture(
        "equation_image",
        _png(image),
        "Equation rendered as a raster image.",
        (
            ExpectedFact("equation", ("s = u t + 1/2 a t^2", "s = ut + 1/2 at^2", "s=ut+1/2at^2")),
            ExpectedFact("a_definition", ("a = acceleration", "a represents acceleration", "a is acceleration")),
        ),
        ("EQUATION OF MOTION", "s = u t + 1/2 a t^2", "where a = acceleration"),
    )

def _decorative_scene() -> VisualFixture:
    image, draw = _canvas()
    draw.rectangle((0, 0, 1000, 380), fill="#85c1e9")
    draw.ellipse((780, 45, 900, 165), fill="#f7dc6f")
    draw.polygon([(0, 430), (250, 180), (480, 430)], fill="#5d6d7e")
    draw.polygon([(300, 430), (610, 130), (900, 430)], fill="#34495e")
    draw.rectangle((0, 430, 1000, 600), fill="#58d68d")
    return VisualFixture(
        "decorative_scene",
        _png(image),
        "Decorative image with no evidentiary role.",
        (
            ExpectedFact("mountains", ("mountain",)),
            ExpectedFact("sun", ("sun",)),
        ),
        (),
    )

def _visual_only_slide() -> VisualFixture:
    image, draw = _canvas()
    draw.rectangle((0, 0, 1000, 600), fill="#f8f9f9")
    draw.text((180, 55), "WHY HYBRID RETRIEVAL?", fill="#1b4f72", font=_font(36))
    draw.ellipse((80, 220, 350, 490), fill="#aed6f1", outline="#2874a6", width=5)
    draw.ellipse((650, 220, 920, 490), fill="#a9dfbf", outline="#239b56", width=5)
    draw.text((145, 330), "KEYWORDS", fill="black", font=_font(25))
    draw.text((705, 330), "MEANING", fill="black", font=_font(25))
    draw.line((350, 355, 650, 355), fill="#7d3c98", width=8)
    draw.polygon([(650, 355), (620, 337), (620, 373)], fill="#7d3c98")
    draw.text((405, 295), "COMBINE", fill="#7d3c98", font=_font(24))
    return VisualFixture(
        "visual_only_slide",
        _png(image),
        "No native slide text was extracted; the image is the only evidence.",
        (
            ExpectedFact("keywords_to_meaning", ("keywords to meaning", "keywords points to meaning", "keywords and meaning")),
            ExpectedFact("combine", ("combine", "combines")),
        ),
        ("WHY HYBRID RETRIEVAL?", "KEYWORDS", "MEANING", "COMBINE"),
    )


def build_labeled_deck() -> tuple[bytes, dict[str, Any]]:
    presentation = Presentation()

    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Native Extraction Baseline — α β γ"
    body = slide.placeholders[1].text_frame
    body.text = "Primary finding: accuracy improved by 12.5%."
    nested = body.add_paragraph()
    nested.text = "Limitation: sample size n = 48."
    nested.level = 1

    table_values = [
        ["Metric", "Baseline", "Proposed"],
        ["F1", "0.71", "0.84"],
        ["Latency (ms)", "120", "95"],
    ]
    table = slide.shapes.add_table(
        3, 3, Inches(0.8), Inches(3.5), Inches(5), Inches(1.5)
    ).table
    for row_index, row in enumerate(table_values):
        for column_index, value in enumerate(row):
            table.cell(row_index, column_index).text = value

    notes = slide.notes_slide.notes_text_frame
    notes.text = "Emphasize that the evaluation is held out."
    notes.add_paragraph().text = "Do not claim causality."

    image_bytes = BytesIO()
    Image.new("RGB", (4, 4), color="blue").save(image_bytes, format="PNG")
    image_bytes.seek(0)
    slide.shapes.add_picture(image_bytes, Inches(6), Inches(4), width=Inches(1))

    second = presentation.slides.add_slide(presentation.slide_layouts[5])
    second.shapes.title.text = "Chart and grouped content"
    group = second.shapes.add_group_shape()
    group.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(3), Inches(0.5)).text = (
        "Grouped method step A → B"
    )

    chart_data = ChartData()
    chart_data.categories = ["Control", "Treatment"]
    chart_data.add_series("Accuracy", (0.71, 0.84))
    chart_shape = second.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(4),
        Inches(1.5),
        Inches(5),
        Inches(3),
        chart_data,
    )
    chart_shape.chart.has_title = True
    chart_shape.chart.chart_title.text_frame.text = "Held-out accuracy"

    buffer = BytesIO()
    presentation.save(buffer)

    expected_blocks = [
        _block("Native Extraction Baseline — α β γ", "title", 1, hierarchy=0),
        _block(
            "Primary finding: accuracy improved by 12.5%.",
            "slide_text",
            1,
            hierarchy=0,
        ),
        _block(
            "Limitation: sample size n = 48.",
            "slide_text",
            1,
            hierarchy=1,
        ),
    ]
    for row_index, row in enumerate(table_values):
        for column_index, value in enumerate(row):
            expected_blocks.append(
                _block(
                    value,
                    "table_cell",
                    1,
                    table_row=row_index,
                    table_column=column_index,
                )
            )
    expected_blocks.extend(
        [
            _block(
                "Emphasize that the evaluation is held out.", "speaker_note", 1
            ),
            _block("Do not claim causality.", "speaker_note", 1),
            _block("Chart and grouped content", "title", 2, hierarchy=0),
            _block("Grouped method step A → B", "slide_text", 2, hierarchy=0),
            _block(
                "Chart title: Held-out accuracy\n\n"
                "| Category | Accuracy |\n"
                "| --- | --- |\n"
                "| Control | 0.71 |\n"
                "| Treatment | 0.84 |",
                "chart_data",
                2,
            ),
        ]
    )

    gold = {
        "blocks": expected_blocks,
        "visual_types": ["picture", "chart"],
        "chart_facts": [
            "Held-out accuracy",
            "Control",
            "Treatment",
            "Accuracy",
            "0.71",
            "0.84",
        ],
    }
    return buffer.getvalue(), gold

def _block(
    text: str,
    block_type: str,
    slide_number: int,
    *,
    hierarchy: int | None = None,
    table_row: int | None = None,
    table_column: int | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "block_type": block_type,
        "slide_number": slide_number,
        "hierarchy_level": hierarchy,
        "table_row": table_row,
        "table_column": table_column,
    }


def build_representative_visual_deck() -> bytes:
    scientific_panel = next(
        fixture
        for fixture in build_visual_fixtures()
        if fixture.fixture_id == "scientific_panel"
    )
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Treatment response overview"
    textbox = slide.shapes.add_textbox(Inches(0.6), Inches(1.1), Inches(8.8), Inches(0.6))
    textbox.text = "Control and treated panels are shown; verify visual counts before reporting."
    slide.shapes.add_picture(
        BytesIO(scientific_panel.image_bytes),
        Inches(0.8),
        Inches(1.8),
        width=Inches(8.4),
    )
    slide.notes_slide.notes_text_frame.text = "The image is synthetic benchmark evidence."
    output = BytesIO()
    presentation.save(output)
    return output.getvalue()


def build_text_pdf(pages: list[list[str]], *, landscape: bool = False) -> bytes:
    """A PDF with one text line per entry, top to bottom, in Helvetica."""
    width, height = (792, 612) if landscape else (612, 792)
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
        body = "BT /F1 11 Tf 14 TL 50 %d Td " % (height - 60) + " ".join(f"({line}) Tj T*" for line in escaped) + " ET"
        objects.append(f"<< /Length {len(body.encode('latin-1'))} >>\nstream\n{body}\nendstream")
        content = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content} 0 R >>")
        kids.append(len(objects))
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode("latin-1")
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1")
    return bytes(out)
