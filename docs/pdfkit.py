"""
Вёрстка PDF «Объяснение проекта» в стиле исходного документа (ReportLab):
заголовки — тёмно-синий шрифт с засечками, подзаголовки — терракотовые, таблицы с тёмной шапкой,
бежевые плашки «для защиты», подписи к рисункам курсивом, колонтитул с номером страницы.
"""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

FONT_DIRS = ["/usr/share/fonts/truetype/liberation2", "/usr/share/fonts/truetype/liberation",
             "C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts/Supplemental"]


def _font(name, files):
    for d in FONT_DIRS:
        for f in files:
            p = os.path.join(d, f)
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont(name, p))
                return
    raise FileNotFoundError(files)


_font("Sans", ["arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"])
_font("Sans-Bold", ["arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf"])
_font("Sans-Italic", ["ariali.ttf", "Arial Italic.ttf", "LiberationSans-Italic.ttf"])
_font("Sans-BoldItalic", ["arialbi.ttf", "Arial Bold Italic.ttf", "LiberationSans-BoldItalic.ttf"])
_font("Serif-Bold", ["georgiab.ttf", "Georgia Bold.ttf", "LiberationSerif-Bold.ttf"])
_font("Mono", ["consola.ttf", "LiberationMono-Regular.ttf"])
FONT_DIRS.append("/usr/share/fonts/truetype/dejavu")
_font("Rub", ["arial.ttf", "DejaVuSans.ttf"])  # знак ₽ есть не во всех шрифтах


def P(t, st):
    return Paragraph(str(t).replace("₽", '<font name="Rub">₽</font>'), st)

pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-BoldItalic")

NAVY = colors.HexColor("#14213D")
RUST = colors.HexColor("#B4501F")
GREY = colors.HexColor("#5B6577")
BEIGE = colors.HexColor("#F4EEE3")
LINE = colors.HexColor("#D9D6CF")

S = {
    "title": ParagraphStyle("title", fontName="Serif-Bold", fontSize=24, leading=29, textColor=NAVY, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=11.5, leading=15, textColor=GREY, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Serif-Bold", fontSize=16.5, leading=21, textColor=NAVY, spaceBefore=12, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.5, leading=15, textColor=RUST, spaceBefore=8, spaceAfter=3, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=10, leading=14.2, spaceAfter=5, alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=10, leading=14.2, spaceAfter=3),
    "caption": ParagraphStyle("caption", fontName="Sans-Italic", fontSize=8.8, leading=12, textColor=GREY, spaceAfter=8),
    "box": ParagraphStyle("box", fontName="Sans", fontSize=10, leading=14.2),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.8, leading=11.5),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8.8, leading=11.5, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=9, leading=12.5, leftIndent=12, spaceAfter=6),
}

W = A4[0] - 2 * 20 * mm  # ширина текста


class Doc:
    def __init__(self):
        self.f = []

    def title(self, t, sub):
        self.f += [P(t, S["title"]), P(sub, S["subtitle"])]

    def h1(self, t):
        self.f.append(P(t, S["h1"]))

    def h2(self, t):
        self.f.append(P(t, S["h2"]))

    def p(self, t):
        self.f.append(P(t, S["body"]))

    def bullets(self, items, numbered=False):
        li = [ListItem(P(x, S["bullet"]), leftIndent=14, value=(i + 1 if numbered else None))
              for i, x in enumerate(items)]
        self.f.append(ListFlowable(li, bulletType="1" if numbered else "bullet", start=1 if numbered else "•",
                                   bulletFontName="Sans-Bold" if numbered else "Sans", bulletFontSize=9 if numbered else 6,
                                   bulletColor=RUST if numbered else colors.black, leftIndent=14))
        self.f.append(Spacer(1, 3))

    def box(self, t):
        tb = Table([[P(t, S["box"])]], colWidths=[W - 8])
        tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BEIGE), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        self.f += [tb, Spacer(1, 7)]

    def table(self, header, rows, widths=None, keep=True):
        data = [[P(str(h), S["cellb"]) for h in header]] + \
               [[P(str(c), S["cell"]) for c in r] for r in rows]
        if widths:
            tot = sum(widths)
            widths = [w / tot * (W - 20) for w in widths]
        tb = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
        tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
        self.f += [KeepTogether([tb]) if keep else tb, Spacer(1, 8)]

    def image(self, path, caption, width=None):
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(path).getSize()
        w = width or W
        self.f.append(KeepTogether([Image(path, width=w, height=w * ih / iw), Spacer(1, 3), P(caption, S["caption"])]))

    def code(self, lines):
        for ln in lines:
            self.f.append(P(ln, S["code"]))

    def br(self):
        self.f.append(PageBreak())

    def build(self, path, footer, title="Восстановление кривой ОФЗ — объяснение проекта"):
        def deco(c, d):
            c.saveState()
            c.setFont("Sans", 8.5); c.setFillColor(GREY)
            if footer:
                c.drawString(20 * mm, 12 * mm, footer)
            c.drawRightString(A4[0] - 20 * mm, 12 * mm, str(d.page))
            c.restoreState()
        doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                                bottomMargin=22 * mm, title=title)
        doc.build(self.f, onFirstPage=deco, onLaterPages=deco)
