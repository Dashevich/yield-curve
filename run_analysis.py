"""
Полный расчёт: кривые всеми методами на каждый торговый день 2024-01 — 2026-10
и сравнение по трём критериям из презентации.

  1. Точность подгонки — RMSE в доходности: в выборке и вне её (leave-one-out).
  2. Длинный конец — ставки 15/20 лет, их дневная волатильность, отклонение от КБД.
  3. Устойчивость при нехватке выпусков — случайно убираем 30% / 50% облигаций
     и смотрим, насколько сдвигается кривая.
  + Форварды как прогноз: 3-месячная ставка через год по форвардной кривой
    против фактической 3-месячной ставки КБД через год.

Результаты: results/*.csv и results/*.png.
Запуск:  python run_analysis.py          (≈5–15 мин, считается параллельно)
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import curves as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GRID = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 12, 15, 20])
METHODS = ["bootstrap", "bootstrap_1y", "ns", "svensson", "kbd"]
LOO_EVERY = 10      # leave-one-out и тест на разреженность — каждый 10-й торговый день
DROP_SHARES = (0.3, 0.5)
DROP_REPS = 10

PERIODS = [  # режимы кривой для агрегирования
    ("2024 H1: высокие ставки", "2024-01-01", "2024-06-30"),
    ("2024 H2 — 2025 H1: инверсия", "2024-07-01", "2025-06-05"),
    ("2025 H2 — 2026 H1: смягчение", "2025-06-06", "2026-06-14"),
    ("Лето 2026: стресс", "2026-06-15", "2026-08-15"),
    ("Осень 2026", "2026-08-16", "2026-12-31"),
]

H = CFS = DAYS = None


def _init():
    global H, CFS, DAYS
    H = C.load_history()
    CFS = C.load_cashflows(H.SECID.unique())
    DAYS = sorted(H.TRADEDATE.unique())


class _KBD:
    def __init__(self, par):
        self.par = par

    def zero(self, t):
        return C.kbd_zero(self.par, t)

    def forward(self, t, h=1e-4):
        t = np.asarray(t, float)
        return ((t + h) * self.zero(t + h) - t * self.zero(t)) / h


def _fwd(f, t1, t2):
    """Форвардная непрерывная ставка между t1 и t2."""
    return float(np.ravel((t2 * f.zero(t2) - t1 * f.zero(t1)) / (t2 - t1))[0])


def _rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def process_day(i):
    d = DAYS[i]
    settle = DAYS[i + 1] if i + 1 < len(DAYS) else d + pd.offsets.BDay(1)
    bonds = C.market_on(H[H.TRADEDATE == d], CFS, settle)
    k = C.load_kbd(d)
    if len(bonds) < 8 or k is None:
        return None
    fits = C.fit_all(bonds)
    fits["kbd"] = _KBD(k[0])
    row = {"date": d, "n_bonds": len(bonds), "max_mat": bonds[-1].maturity,
           "volume_bn": sum(b.volume for b in bonds) / 1e9}
    curves_rows, err_rows = [], []
    for m, f in fits.items():
        e = C.yield_errors(bonds, f.zero)
        row[f"rmse_{m}"] = _rmse(e)
        row[f"mae_{m}"] = float(np.mean(np.abs(e)))
        z = f.zero(GRID)
        curves_rows.append({"date": d, "method": m, **{f"z{g:g}": C.to_annual(v) for g, v in zip(GRID, z)},
                            "fwd_1y_3m": C.to_annual(_fwd(f, 1.0, 1.25)),
                            "fwd_15_20": C.to_annual(_fwd(f, 15.0, 20.0))})
        for b, ee in zip(bonds, e):
            err_rows.append({"date": d, "method": m, "secid": b.secid, "maturity": b.maturity,
                             "coupon_rate": b.cf[0] / 1000 * 2 if len(b.cf) > 1 else np.nan, "err_bp": ee})
    if "bootstrap" in fits:
        row["bootstrap_fwd_min"] = C.to_annual(np.min(fits["bootstrap"].fwd))
        row["bootstrap_fwd_max"] = C.to_annual(np.max(fits["bootstrap"].fwd))

    loo_rows, drop_rows = [], []
    if i % LOO_EVERY == 0:
        warm = {"ns": fits["ns"].params, "svensson": fits["svensson"].params}
        # --- leave-one-out: оцениваем каждую облигацию кривой, построенной без неё
        for j, b in enumerate(bonds):
            rest = bonds[:j] + bonds[j + 1:]
            sub = {
                "bootstrap": C.Bootstrap(rest),
                "bootstrap_1y": C.Bootstrap(rest, bucket=1.0),
                "ns": C.Parametric(rest, "ns", warm["ns"], only_x0=True),
                "svensson": C.Parametric(rest, "sv", warm["svensson"], only_x0=True),
            }
            for m, f in sub.items():
                loo_rows.append({"date": d, "method": m, "secid": b.secid, "maturity": b.maturity,
                                 "err_bp": float(C.yield_errors([b], f.zero)[0])})
        # --- разреженность: случайно выкидываем долю выпусков
        rng = np.random.default_rng(i)
        g = np.linspace(1, 15, 29)
        for share in DROP_SHARES:
            for rep in range(DROP_REPS):
                keep = np.sort(rng.choice(len(bonds), int(round(len(bonds) * (1 - share))), replace=False))
                rest = [bonds[j] for j in keep]
                left = [b for j, b in enumerate(bonds) if j not in set(keep)]
                sub = {
                    "bootstrap": C.Bootstrap(rest),
                    "bootstrap_1y": C.Bootstrap(rest, bucket=1.0),
                    "ns": C.Parametric(rest, "ns", warm["ns"], only_x0=True),
                    "svensson": C.Parametric(rest, "sv", warm["svensson"], only_x0=True),
                }
                for m, f in sub.items():
                    shift = (f.zero(g) - fits[m].zero(g)) * 1e4
                    drop_rows.append({"date": d, "method": m, "share": share, "rep": rep,
                                      "curve_shift_bp": _rmse(shift),
                                      "shift_15y_bp": float(np.ravel(f.zero(15.0) - fits[m].zero(15.0))[0] * 1e4),
                                      "oos_rmse_bp": _rmse(C.yield_errors(left, f.zero))})
    return row, curves_rows, err_rows, loo_rows, drop_rows


def main():
    os.makedirs(OUT, exist_ok=True)
    _init()
    t0 = time.time()
    rows, curves_rows, err_rows, loo_rows, drop_rows = [], [], [], [], []
    with Pool(initializer=_init) as pool:
        for n, r in enumerate(pool.imap(process_day, range(len(DAYS)), chunksize=4), 1):
            if r is not None:
                rows.append(r[0]); curves_rows += r[1]; err_rows += r[2]; loo_rows += r[3]; drop_rows += r[4]
            if n % 50 == 0 or n == len(DAYS):
                print(f"  {n}/{len(DAYS)} дней, {time.time() - t0:.0f} c", flush=True)
    daily = pd.DataFrame(rows)
    curves = pd.DataFrame(curves_rows)
    errs = pd.DataFrame(err_rows)
    loo = pd.DataFrame(loo_rows)
    drop = pd.DataFrame(drop_rows)
    for name, df in [("daily_fit", daily), ("curves", curves), ("bond_errors", errs),
                     ("loo_errors", loo), ("sparsity", drop)]:
        df.to_csv(os.path.join(OUT, name + ".csv"), index=False, encoding="utf-8-sig")
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин. Файлы в {OUT}")


if __name__ == "__main__":
    sys.exit(main())
