"""
Build the APA-style academic report as a PDF using reportlab.

Mirrors the Word version: title page, abstract with keywords, level-1/level-2
headings, numbered figures with italic captions, APA tables, and a
hanging-indent reference list. A running head and page number appear on every
page. Wording is imported from report_content so Word and PDF match exactly.

Run:  python report_build_pdf.py
"""
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)
from reportlab.lib.utils import ImageReader

import report_content as RC

OUT = Path(__file__).resolve().parent / "reports" / \
    "AAI-540_Heart_Disease_Screening_APA_Report.pdf"

FONT = "Times-Roman"
FONT_B = "Times-Bold"
FONT_I = "Times-Italic"


def _styles():
    ss = getSampleStyleSheet()
    body = ParagraphStyle(
        "APABody", parent=ss["Normal"], fontName=FONT, fontSize=12,
        leading=24, firstLineIndent=0.5 * inch, alignment=TA_LEFT,
        spaceAfter=0)
    body_noindent = ParagraphStyle(
        "APABodyNI", parent=body, firstLineIndent=0)
    h1 = ParagraphStyle(
        "APAH1", parent=body, fontName=FONT_B, alignment=TA_CENTER,
        firstLineIndent=0, spaceBefore=12, spaceAfter=6)
    h2 = ParagraphStyle(
        "APAH2", parent=body, fontName=FONT_B, alignment=TA_LEFT,
        firstLineIndent=0, spaceBefore=10, spaceAfter=4)
    title = ParagraphStyle(
        "APATitle", parent=body, fontName=FONT_B, alignment=TA_CENTER,
        firstLineIndent=0, fontSize=14, leading=20)
    center = ParagraphStyle(
        "APACenter", parent=body, alignment=TA_CENTER, firstLineIndent=0)
    cap_num = ParagraphStyle(
        "CapNum", parent=body, fontName=FONT_B, firstLineIndent=0,
        leading=14, spaceBefore=6)
    cap_ttl = ParagraphStyle(
        "CapTtl", parent=body, fontName=FONT_I, firstLineIndent=0,
        leading=14, spaceAfter=4)
    ref = ParagraphStyle(
        "APARef", parent=body, firstLineIndent=-0.5 * inch,
        leftIndent=0.5 * inch)
    kw = ParagraphStyle(
        "APAKw", parent=body_noindent)
    return dict(body=body, body_ni=body_noindent, h1=h1, h2=h2, title=title,
                center=center, cap_num=cap_num, cap_ttl=cap_ttl, ref=ref, kw=kw)


def _page_furniture(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 12)
    canvas.drawString(inch, letter[1] - 0.6 * inch, RC.RUNNING_HEAD)
    canvas.drawRightString(letter[0] - inch, letter[1] - 0.6 * inch,
                           str(doc.page))
    canvas.restoreState()


def build():
    S = _styles()
    doc = BaseDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=_page_furniture)])

    story = []

    # --- Title page ---
    story.append(Spacer(1, 2.2 * inch))
    story.append(Paragraph(RC.TITLE, S["title"]))
    story.append(Spacer(1, 0.4 * inch))
    for line in (RC.AUTHORS, RC.UNIVERSITY, RC.COURSE,
                 f"Instructor: {RC.INSTRUCTOR}", RC.DATE):
        story.append(Paragraph(line, S["center"]))
    story.append(PageBreak())

    # --- Abstract ---
    story.append(Paragraph("Abstract", S["h1"]))
    story.append(Paragraph(RC.ABSTRACT, S["body_ni"]))
    story.append(Paragraph(
        f'<i>Keywords:</i> {RC.KEYWORDS}', S["kw"]))
    story.append(PageBreak())

    # --- Body ---
    story.append(Paragraph(RC.TITLE, S["h1"]))
    for level, heading, body, figs, tables in RC.SECTIONS:
        story.append(Paragraph(heading, S["h1"] if level == 1 else S["h2"]))
        for para in body:
            story.append(Paragraph(para, S["body"]))
        for spec in tables:
            _add_table(story, spec, S)
        for path, cap in figs:
            _add_figure(story, path, cap, S)

    # --- References ---
    story.append(PageBreak())
    story.append(Paragraph("References", S["h1"]))
    for r in RC.REFERENCES:
        story.append(Paragraph(r, S["ref"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    print("Wrote", OUT)


def _add_figure(story, path, caption, S):
    if not Path(path).exists():
        return
    parts = caption.split(". ", 1)
    story.append(Paragraph(parts[0] + ("." if len(parts) > 1 else ""),
                           S["cap_num"]))
    if len(parts) > 1:
        story.append(Paragraph(parts[1], S["cap_ttl"]))
    ir = ImageReader(str(path))
    iw, ih = ir.getSize()
    w = 5.6 * inch
    h = w * ih / iw
    max_h = 6.5 * inch
    if h > max_h:
        h = max_h
        w = h * iw / ih
    img = Image(str(path), width=w, height=h)
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Spacer(1, 0.2 * inch))


def _add_table(story, spec, S):
    caption, header, rows = spec
    parts = caption.split("\n", 1)
    story.append(Paragraph(parts[0], S["cap_num"]))
    if len(parts) > 1:
        story.append(Paragraph(parts[1], S["cap_ttl"]))
    data = [header] + rows
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTNAME", (0, 0), (-1, 0), FONT_B),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.2 * inch))


if __name__ == "__main__":
    build()
