"""
Внутридневные 10-минутные свечи по ОФЗ-ПД с Мосбиржи — для «точного» анализа стрессовых дней.

Какие дни (выбор воспроизводимый, по results/market.csv, т.е. после run_analysis.py):
  * все дни заседаний Банка России по ключевой ставке внутри периода (решение — в 13:30 МСК);
  * 30 стрессовых дней с самым большим движением 10-летней ставки КБД;
  * 30 случайных спокойных дней (контроль, seed=0).

Куда: data/raw/candles/<дата>/<SECID>.csv (open, close, high, low, value, volume, begin, end).
Время в свечах — московское.

Запуск:  python download_intraday.py         (≈10 мин; уже скачанное пропускается)
"""

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import download_data as D
import curves as C

OUT = os.path.join(D.ROOT, "candles")
INTERVAL = 10
N_STRESS = 30
N_CALM = 30


def select_days():
    """{дата: тип дня}."""
    m = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "market.csv"),
                    parse_dates=["date"], encoding="utf-8-sig").set_index("date")
    days = list(m.index)
    out = {}
    for d in C.load_cbr_meetings():
        if days[0] <= d <= days[-1] and d in m.index:
            out[d] = "Заседание ЦБ"
    move = m.kbd10.diff().abs()
    st = move[(m.regime == "Стресс") & ~move.index.isin(list(out))].dropna()
    for d in st.sort_values(ascending=False).index[:N_STRESS]:
        out[d] = "Стресс"
    calm = m.index[(m.regime == "Спокойно") & ~m.index.isin(list(out))]
    for d in pd.Series(calm).sample(N_CALM, random_state=0):
        out[d] = "Спокойно"
    return dict(sorted(out.items()))


def secids_on(h, d):
    return sorted(h[(h.TRADEDATE == d) & (h.NUMTRADES.fillna(0) >= 5)].SECID.unique())


def fetch(task):
    d, s = task
    p = os.path.join(OUT, d.strftime("%Y-%m-%d"), s + ".csv")
    if os.path.exists(p):
        return 0
    ds = d.strftime("%Y-%m-%d")
    try:
        cols, rows = D.fetch_block(f"engines/stock/markets/bonds/boards/TQOB/securities/{s}/candles.json",
                                   {"interval": INTERVAL, "from": ds, "till": ds}, "candles")
    except Exception as e:
        D.log_error(f"candles {s} {ds}: {e}")
        return 0
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols or ["open", "close", "high", "low", "value", "volume", "begin", "end"])
        w.writerows(rows)
    os.replace(p + ".tmp", p)
    return 1


def main():
    days = select_days()
    os.makedirs(OUT, exist_ok=True)
    pd.Series({d.strftime("%Y-%m-%d"): t for d, t in days.items()}, name="kind").rename_axis("date") \
        .to_csv(os.path.join(OUT, "days.csv"), encoding="utf-8")
    print({t: sum(v == t for v in days.values()) for t in set(days.values())})
    h = C.load_history()
    tasks = [(d, s) for d in days for s in secids_on(h, d)]
    print(f"{len(tasks)} запросов")
    done = 0
    with ThreadPoolExecutor(4) as ex:
        for i, n in enumerate(ex.map(fetch, tasks), 1):
            done += n
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    print(f"Готово, скачано {done} файлов -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
