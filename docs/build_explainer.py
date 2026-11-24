"""
Собирает docs/Объяснение_проекта.pdf — объяснение проекта «с нуля» (понятия, данные, методы,
результаты, стресс-анализ, слабые места, вопросы на защите).

Числа берутся из results/numbers.json (его пишет report.py), графики — из results/fig_*.png.
Запуск (после run_analysis.py и report.py):  python docs/build_explainer.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pdfkit as K  # noqa: E402

RES = os.path.join(HERE, "..", "results")
N = json.load(open(os.path.join(RES, "numbers.json"), encoding="utf-8"))
FIG = lambda name: os.path.join(RES, name)

NM = {"bootstrap": "Бутстрэп (все выпуски)", "bootstrap_1y": "Бутстрэп (1 выпуск/год)", "ns": "Нельсон–Сигель",
      "svensson": "Свенссон", "ns_fix": "NS с фикс. τ", "ns_anchor": "NS + якорь RUONIA",
      "spline": "Сглаживающий сплайн", "kbd": "КБД Мосбиржи"}
OWN = ["bootstrap", "bootstrap_1y", "ns", "svensson", "ns_fix", "ns_anchor", "spline"]
REG = ["Спокойно", "Обычно", "Стресс"]


def f1(x):
    return "—" if x is None else f"{x:.1f}".replace(".", ",")


def f0(x):
    return "—" if x is None else f"{x:.0f}"


def pct(x):
    return f"{x * 100:.0f}%"


def build():
    import content
    d = K.Doc()
    content.write(d, N, FIG, f1, f0, pct, NM, OWN, REG)
    out = os.path.join(HERE, "Объяснение_проекта.pdf")
    d.build(out, "Кривая ОФЗ: объяснение проекта")
    print("PDF:", out)


if __name__ == "__main__":
    build()
