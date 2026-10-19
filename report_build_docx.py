"""
Build the APA-style academic report as a Word (.docx) document.

Implements APA-7 student-paper conventions: a title page, a running head in the
page header, an abstract with keywords, double-spaced 12-pt Times New Roman body
text, level-1 (centered, bold) and level-2 (flush-left, bold) headings, numbered
figures with italic captions, APA-style tables, and a hanging-indent reference
list.

Run:  python report_build_docx.py
"""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

import report_content as RC

OUT = Path(__file__).resolve().parent / "reports" / \
    "AAI-540_Heart_Disease_Screening_APA_Report.docx"


def _set_base_style(doc):
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    pf = style.paragraph_format
    pf.line_spacing = 2.0
    pf.space_after = Pt(0)
    pf.space_before = Pt(0)


def _running_head(doc):
    section = doc.sections[0]
    header = section.header
    p = header.paragraphs[0]
    p.text = RC.RUNNING_HEAD
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in p.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)


def _para(doc, text, align=WD_ALIGN_PARAGRAPH.LEFT, bold=False, italic=False,
          indent_first=True, spacing=2.0):
    p = doc.add_paragraph()
    p.alignment = align
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    p.paragraph_format.line_spacing = spacing
    if indent_first and align == WD_ALIGN_PARAGRAPH.LEFT:
        p.paragraph_format.first_line_indent = Inches(0.5)
    return p


def _title_page(doc):
    for _ in range(3):
        doc.add_paragraph()
    _para(doc, RC.TITLE, WD_ALIGN_PARAGRAPH.CENTER, bold=True,
          indent_first=False)
    doc.add_paragraph()
    for line in (RC.AUTHORS, f"{RC.UNIVERSITY}", RC.COURSE,
                 f"Instructor: {RC.INSTRUCTOR}", RC.DATE):
        _para(doc, line, WD_ALIGN_PARAGRAPH.CENTER, indent_first=False)
    doc.add_page_break()


def _abstract(doc):
    _para(doc, "Abstract", WD_ALIGN_PARAGRAPH.CENTER, bold=True,
          indent_first=False)
    _para(doc, RC.ABSTRACT, indent_first=False)
    kw = doc.add_paragraph()
    kw.paragraph_format.line_spacing = 2.0
    r1 = kw.add_run("Keywords: ")
    r1.italic = True
    r1.font.name = "Times New Roman"; r1.font.size = Pt(12)
    r2 = kw.add_run(RC.KEYWORDS)
    r2.font.name = "Times New Roman"; r2.font.size = Pt(12)
    doc.add_page_break()


def _heading(doc, text, level):
    align = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
    _para(doc, text, align, bold=True, indent_first=False)


def _figure(doc, path, caption):
    if not Path(path).exists():
        return
    # APA figure: number+title above, image, then a note is optional.
    num_title = caption.split(". ", 1)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(num_title[0] if len(num_title) > 1 else caption)
    r.bold = True
    r.font.name = "Times New Roman"; r.font.size = Pt(12)
    if len(num_title) > 1:
        pt = doc.add_paragraph()
        pt.paragraph_format.line_spacing = 1.0
        rt = pt.add_run(num_title[1])
        rt.italic = True
        rt.font.name = "Times New Roman"; rt.font.size = Pt(12)
    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.line_spacing = 1.0
    pic.add_run().add_picture(str(path), width=Inches(5.8))
    doc.add_paragraph()


def _table(doc, spec):
    caption, header, rows = spec
    parts = caption.split("\n", 1)
    # Table number (bold) then italic title.
    p = doc.add_paragraph(); p.paragraph_format.line_spacing = 1.0
    r = p.add_run(parts[0]); r.bold = True
    r.font.name = "Times New Roman"; r.font.size = Pt(12)
    if len(parts) > 1:
        pt = doc.add_paragraph(); pt.paragraph_format.line_spacing = 1.0
        rt = pt.add_run(parts[1]); rt.italic = True
        rt.font.name = "Times New Roman"; rt.font.size = Pt(12)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light List Accent 1"
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = h
        for para in c.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.size = Pt(11)
                run.font.name = "Times New Roman"
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = str(val)
            for para in cells[j].paragraphs:
                for run in para.runs:
                    run.font.size = Pt(11)
                    run.font.name = "Times New Roman"
    doc.add_paragraph()


def _references(doc):
    doc.add_page_break()
    _heading(doc, "References", 1)
    for ref in RC.REFERENCES:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 2.0
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.5)  # hanging indent
        r = p.add_run(ref)
        r.font.name = "Times New Roman"; r.font.size = Pt(12)


def build():
    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(1)
        s.left_margin = s.right_margin = Inches(1)
    _set_base_style(doc)
    _running_head(doc)
    _title_page(doc)
    _abstract(doc)

    # Body opens by repeating the title (APA convention).
    _para(doc, RC.TITLE, WD_ALIGN_PARAGRAPH.CENTER, bold=True,
          indent_first=False)

    for level, heading, body, figs, tables in RC.SECTIONS:
        _heading(doc, heading, level)
        for para in body:
            _para(doc, para)
        for spec in tables:
            _table(doc, spec)
        for path, cap in figs:
            _figure(doc, path, cap)

    _references(doc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
