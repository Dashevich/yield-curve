"""
Подбор параметров штрафа сглаживающего сплайна (curves.Spline: L, S) по leave-one-out.

Подбираем только на 2024 H1 — до периода стресса, чтобы не подгонять метод
под те данные, на которых его потом сравниваем.
Запуск:  python tune_spline.py
"""

import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd

import curves as C

GRID_L = [-8, -6, -5, -4, -3, -2, 0, 2]
GRID_S = [-12, -10, -8, -6, -5, -4, -3]
H = CFS = DAYS = None


def _init():
    global H, CFS, DAYS
    H = C.load_history()
    CFS = C.load_cashflows(H.SECID.unique())
    all_days = sorted(H.TRADEDATE.unique())
    DAYS = [(d, all_days[i + 1]) for i, d in enumerate(all_days[:-1])
            if d <= pd.Timestamp("2024-06-30") and i % 10 == 0]


def loo_day(k):
    d, settle = DAYS[k]
    bonds = C.market_on(H[H.TRADEDATE == d], CFS, settle)
    out = {}
    for L, S in itertools.product(GRID_L, GRID_S):
        if S > L:
            continue
        full = C.Spline(bonds, L=L, S=S)
        e = [C.yield_errors([b], C.Spline(bonds[:j] + bonds[j + 1:], full.params, L=L, S=S).zero)[0]
             for j, b in enumerate(bonds)]
        out[(L, S)] = e
    return out


def main():
    _init()
    with Pool(initializer=_init) as pool:
        res = pool.map(loo_day, range(len(DAYS)))
    rows = []
    for key in res[0]:
        e = np.concatenate([r[key] for r in res])
        rows.append({"L": key[0], "S": key[1], "loo_rmse_bp": float(np.sqrt(np.mean(e ** 2)))})
    t = pd.DataFrame(rows).pivot(index="L", columns="S", values="loo_rmse_bp")
    print(f"LOO RMSE, б.п., {len(DAYS)} дней 2024 H1 (строки — L, столбцы — S):")
    print(t.round(1).to_string())
    b = min(rows, key=lambda r: r["loo_rmse_bp"])
    print(f"Лучшие: L={b['L']}, S={b['S']}, RMSE={b['loo_rmse_bp']:.1f}")


if __name__ == "__main__":
    main()
