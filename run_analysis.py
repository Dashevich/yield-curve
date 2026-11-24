"""
Полный расчёт: кривые всеми методами на каждый торговый день 2024-01 — 2026-10
и сравнение по критериям из презентации + стресс-анализ.

  1. Точность подгонки — RMSE в доходности: в выборке и вне её (leave-one-out, КАЖДЫЙ день).
  2. Длинный конец — ставки 10/20 лет, их дневная волатильность, отклонение от КБД.
  3. Устойчивость при нехватке выпусков:
       a) случайно убираем 30% / 50% облигаций;
       b) «как в стресс»: убираем весь длинный конец (> 10 лет), оставляем 12 самых ликвидных,
          убираем короткий конец (< 2 лет).
  4. Шум цен: возмущаем цены на величину оценённого спреда бид/аск (Corwin–Schultz по HIGH/LOW).
  5. Модельный риск: цена одних и тех же «синтетических» облигаций по разным кривым.
  + Форварды как прогноз (анализ — в report.py).

Рыночный режим каждого дня (спокойно / обычно / стресс) определяется по данным, а не по
календарю: волатильность 10-летней ставки КБД и просадка индекса RGBI (см. market_regimes).
Тяжёлые проверки (3, 4) — на каждом 5-м дне и на ВСЕХ стрессовых днях.

Результаты: results/*.csv.
Запуск:  python run_analysis.py            (≈10–20 мин на 8 ядрах)
         python run_analysis.py --tune     (подбор τ для NS-fix и штрафа сплайна на 2024 H1, ≈3 мин)
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
OWN = C.METHODS                                  # собственные методы
METHODS = OWN + ["kbd"]
WORKERS = int(os.environ.get("WORKERS", min(os.cpu_count() or 1, 6)))
HEAVY_EVERY = 5     # случайная разреженность, сценарии и шум — каждый 5-й день + все стрессовые
DROP_SHARES = (0.3, 0.5)
DROP_REPS = 10
NOISE_REPS = 20
SYNTH = (2, 5, 10, 15, 20)  # сроки «синтетических» облигаций (купон 12%, раз в полгода)
SYNTH_COUPON = 0.12

PERIODS = [  # календарные режимы кривой (как в исходной версии)
    ("2024 H1: высокие ставки", "2024-01-01", "2024-06-30"),
    ("2024 H2 — 2025 H1: инверсия", "2024-07-01", "2025-06-05"),
    ("2025 H2 — 2026 H1: смягчение", "2025-06-06", "2026-06-14"),
    ("Лето 2026: стресс", "2026-06-15", "2026-08-15"),
    ("Осень 2026", "2026-08-16", "2026-12-31"),
]
REGIMES = ["Спокойно", "Обычно", "Стресс"]
VOL_WINDOW = 20      # дней для скользящей волатильности
DD_WINDOW = 63       # дней (≈ квартал) для просадки RGBI
STRESS_VOL_Q = 0.80  # стресс: волатильность выше 80% дней ...
STRESS_DD = -0.07    # ... или RGBI ниже квартального максимума больше чем на 7% (худшие ~10% дней)
CALM_VOL_Q = 0.50    # спокойно: волатильность ниже медианы и просадка меньше 3%
CALM_DD = -0.03

H = CFS = DAYS = ISSUE = MKT = RUONIA = None


def add_spreads(h):
    """Оценка спреда бид/аск по максимуму и минимуму цены за 2 дня (Corwin & Schultz, 2012).
    Котировок бид/аск в итогах торгов нет; это стандартная замена. Колонка HALF_SPREAD — полуспред
    в долях цены. Если оценка не получилась — (HIGH−LOW)/4 от цены, в крайнем случае медиана дня."""
    h = h.sort_values(["SECID", "TRADEDATE"]).copy()
    hi, lo = h.HIGH.astype(float), h.LOW.astype(float)
    ok = (hi > 0) & (lo > 0) & (hi >= lo)
    hi, lo = hi.where(ok), lo.where(ok)
    hi1, lo1 = hi.groupby(h.SECID).shift(1), lo.groupby(h.SECID).shift(1)
    beta = np.log(hi / lo) ** 2 + np.log(hi1 / lo1) ** 2
    gamma = np.log(np.maximum(hi, hi1) / np.minimum(lo, lo1)) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0)
    s = s.where(s > 0, (hi - lo) / ((hi + lo) / 2) / 2)  # полный спред ≈ половина дневного диапазона
    hs = (s / 2).clip(upper=0.01)
    hs = hs.fillna(hs.groupby(h.TRADEDATE).transform("median"))
    h["HALF_SPREAD"] = hs.fillna(0.0005)
    return h


def market_regimes(h, days):
    """Рыночные индикаторы и режим каждого дня."""
    rows = []
    for d in days:
        k = C.load_kbd(d)
        hd = h[h.TRADEDATE == d]
        rows.append({"date": d,
                     "kbd1": C.to_annual(C.kbd_zero(k[0], 1.0)) if k else np.nan,
                     "kbd10": C.to_annual(C.kbd_zero(k[0], 10.0)) if k else np.nan,
                     "turnover_bn": hd.VALUE.fillna(0).sum() / 1e9,
                     "n_traded": int((hd.NUMTRADES.fillna(0) >= 5).sum())})
    m = pd.DataFrame(rows).set_index("date")
    m["vol20_bp"] = m.kbd10.diff().mul(100).rolling(VOL_WINDOW, min_periods=10).std()
    rg = C.load_rgbi()
    if rg is not None:
        m["rgbi"] = rg.reindex(m.index).ffill()
        m["drawdown"] = m.rgbi / m.rgbi.rolling(DD_WINDOW, min_periods=5).max() - 1
    else:
        m["rgbi"] = np.nan
        m["drawdown"] = 0.0
    ru = C.load_cbr_rate("ruonia")
    kr = C.load_cbr_rate("keyrate")
    m["ruonia"] = ru.reindex(m.index).values if ru is not None else np.nan
    m["keyrate"] = kr.reindex(m.index).values if kr is not None else np.nan
    v_hi, v_lo = m.vol20_bp.quantile(STRESS_VOL_Q), m.vol20_bp.quantile(CALM_VOL_Q)
    stress = (m.vol20_bp >= v_hi) | (m.drawdown <= STRESS_DD)
    calm = (m.vol20_bp <= v_lo) & (m.drawdown > CALM_DD)
    m["regime"] = np.where(stress, "Стресс", np.where(calm, "Спокойно", "Обычно"))
    m.attrs.update(vol_hi=v_hi, vol_lo=v_lo)
    return m


def _init():
    global H, CFS, DAYS, ISSUE, MKT
    H = add_spreads(C.load_history())
    CFS = C.load_cashflows(H.SECID.unique())
    ISSUE = C.load_issue_dates(H.SECID.unique())
    DAYS = sorted(H.TRADEDATE.unique())
    p = os.path.join(OUT, "market.csv")
    MKT = pd.read_csv(p, parse_dates=["date"], encoding="utf-8-sig").set_index("date") if os.path.exists(p) else None


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
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def _synth_price(f, T):
    t = np.arange(T, 0, -0.5)[::-1]
    cf = np.full(len(t), 1000 * SYNTH_COUPON / 2)
    cf[-1] += 1000
    return float(np.sum(cf * np.exp(-np.ravel(f.zero(t)) * t)))


def _warm(fits):
    return {m: f.params for m, f in fits.items() if hasattr(f, "params")}


def _refit(rest, warm, short_rate):
    """Кривые на подвыборке. Для параметрических — старт с решения на полной выборке + мульти-старт."""
    out = {}
    for m in OWN:
        try:
            f = C.fit_one(m, rest, warm.get(m), only_warm=False, short_rate=short_rate)
        except Exception:
            f = None
        if f is not None:
            out[m] = f
    return out


def process_day(i):
    d = DAYS[i]
    settle = DAYS[i + 1] if i + 1 < len(DAYS) else d + pd.offsets.BDay(1)
    bonds = C.market_on(H[H.TRADEDATE == d], CFS, settle, issue_dates=ISSUE)
    k = C.load_kbd(d)
    if len(bonds) < 8 or k is None:
        return None
    mk = MKT.loc[d] if MKT is not None and d in MKT.index else None
    regime = mk.regime if mk is not None else "Обычно"
    ru = float(C.simple_to_cont(mk.ruonia)) if mk is not None and pd.notna(mk.ruonia) else None
    fits = C.fit_all(bonds, short_rate=ru)
    fits["kbd"] = _KBD(k[0])
    row = {"date": d, "regime": regime, "n_bonds": len(bonds), "max_mat": bonds[-1].maturity,
           "volume_bn": sum(b.volume for b in bonds) / 1e9,
           "half_spread_bp": float(np.nanmedian([b.half_spread / b.dirty / b.dur * 1e4 for b in bonds])),
           "bootstrap_skipped": fits["bootstrap"].skipped}
    for m in ("ns", "ns_robust", "svensson", "ns_fix", "ns_anchor"):
        if m in fits:
            row[f"{m}_at_bound"] = fits[m].at_bound
            row[f"{m}_short_active"] = fits[m].short_active
            row[f"{m}_params"] = " ".join(f"{x:.6g}" for x in fits[m].params)
    curves_rows, err_rows = [], []
    for m, f in fits.items():
        e = C.yield_errors(bonds, f.zero)
        row[f"rmse_{m}"] = _rmse(e)
        row[f"mae_{m}"] = float(np.mean(np.abs(e)))
        z = np.ravel(f.zero(GRID))
        curves_rows.append({"date": d, "method": m, **{f"z{g:g}": C.to_annual(v) for g, v in zip(GRID, z)},
                            "fwd_1y_3m": C.to_annual(_fwd(f, 1.0, 1.25)),
                            "fwd_15_20": C.to_annual(_fwd(f, 15.0, 20.0)),
                            **{f"px{T}": _synth_price(f, T) for T in SYNTH}})
        for b, ee in zip(bonds, e):
            err_rows.append({"date": d, "method": m, "secid": b.secid, "maturity": b.maturity, "dur": b.dur,
                             "coupon_rate": b.coupon, "volume": b.volume, "age": b.age, "err_bp": ee})
    row["bootstrap_fwd_min"] = C.to_annual(np.min(fits["bootstrap"].fwd))
    row["bootstrap_fwd_max"] = C.to_annual(np.max(fits["bootstrap"].fwd))
    warm = _warm(fits)

    # --- leave-one-out: каждую облигацию оцениваем кривой, построенной без неё (каждый день)
    loo_rows = []
    for j, b in enumerate(bonds):
        rest = bonds[:j] + bonds[j + 1:]
        for m, f in _refit(rest, warm, ru).items():
            loo_rows.append({"date": d, "method": m, "secid": b.secid, "maturity": b.maturity,
                             "err_bp": float(C.yield_errors([b], f.zero)[0])})

    drop_rows, scen_rows, noise_rows = [], [], []
    if i % HEAVY_EVERY == 0 or regime == "Стресс":
        rng = np.random.default_rng(i)
        g = np.linspace(1, 15, 29)
        base = {m: np.ravel(fits[m].zero(g)) for m in OWN if m in fits}

        def shift(f, m, grid=g, ref=None):
            ref = base[m] if ref is None else ref
            return _rmse((np.ravel(f.zero(grid)) - ref) * 1e4)

        # --- a) случайная разреженность
        for share in DROP_SHARES:
            for rep in range(DROP_REPS):
                keep = set(rng.choice(len(bonds), int(round(len(bonds) * (1 - share))), replace=False).tolist())
                rest = [b for j, b in enumerate(bonds) if j in keep]
                left = [b for j, b in enumerate(bonds) if j not in keep]
                for m, f in _refit(rest, warm, ru).items():
                    drop_rows.append({"date": d, "regime": regime, "method": m, "share": share, "rep": rep,
                                      "curve_shift_bp": shift(f, m),
                                      "shift_15y_bp": float(np.ravel(f.zero(15.0) - fits[m].zero(15.0))[0] * 1e4),
                                      "oos_rmse_bp": _rmse(C.yield_errors(left, f.zero))})
        # --- b) структурные сценарии «как в стресс»
        top = set(sorted(range(len(bonds)), key=lambda j: -bonds[j].volume)[:12])
        scen = {"Без длинных (> 10 лет)": [j for j, b in enumerate(bonds) if b.maturity <= 10],
                "12 самых ликвидных": sorted(top),
                "Без коротких (< 2 лет)": [j for j, b in enumerate(bonds) if b.maturity >= 2]}
        for name, keep in scen.items():
            keep = set(keep)
            rest = [b for j, b in enumerate(bonds) if j in keep]
            left = [b for j, b in enumerate(bonds) if j not in keep]
            if len(rest) < 6 or not left:
                continue
            for m, f in _refit(rest, warm, ru).items():
                scen_rows.append({"date": d, "regime": regime, "method": m, "scenario": name,
                                  "curve_shift_bp": shift(f, m),
                                  "shift_20y_bp": float(np.ravel(f.zero(20.0) - fits[m].zero(20.0))[0] * 1e4),
                                  "oos_rmse_bp": _rmse(C.yield_errors(left, f.zero))})
        # --- c) шум цен в пределах спреда
        for rep in range(NOISE_REPS):
            noisy = []
            for b in bonds:
                try:
                    noisy.append(C.with_price(b, b.dirty + rng.normal(0, b.half_spread)))
                except ValueError:
                    noisy.append(b)
            for m, f in _refit(noisy, warm, ru).items():
                noise_rows.append({"date": d, "regime": regime, "method": m, "rep": rep,
                                   "curve_shift_bp": shift(f, m),
                                   "shift_10y_bp": float(np.ravel(f.zero(10.0) - fits[m].zero(10.0))[0] * 1e4),
                                   "shift_20y_bp": float(np.ravel(f.zero(20.0) - fits[m].zero(20.0))[0] * 1e4)})
    return row, curves_rows, err_rows, loo_rows, drop_rows, scen_rows, noise_rows


# ---------------------------------------------------------------- подбор гиперпараметров
def _tune_day(i):
    d = DAYS[i]
    settle = DAYS[i + 1] if i + 1 < len(DAYS) else d + pd.offsets.BDay(1)
    bonds = C.market_on(H[H.TRADEDATE == d], CFS, settle)
    if len(bonds) < 8:
        return None
    tau = C.Parametric(bonds, "ns").params[3]
    res = {"date": d, "tau": tau}
    for lam in TUNE_LAMS:
        e = []
        for j, b in enumerate(bonds):
            f = C.Spline(bonds[:j] + bonds[j + 1:], lam=lam)
            e.append(C.yield_errors([b], f.zero)[0])
        res[f"lam_{lam:g}"] = _rmse(e)
    return res


TUNE_LAMS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


TUNE_END = "2024-06-30"  # подбираем только на 2024 H1 — до основных стрессовых эпизодов и вне основной проверки


def tune():
    _init()
    idx = [i for i, d in enumerate(DAYS) if d <= pd.Timestamp(TUNE_END)][::3]
    with Pool(initializer=_init) as pool:
        r = [x for x in pool.map(_tune_day, idx) if x is not None]
    t = pd.DataFrame(r)
    print(f"Медиана τ свободного NS: {t.tau.median():.3f} лет (квартили {t.tau.quantile(.25):.2f}–{t.tau.quantile(.75):.2f})")
    print("RMSE вне выборки сплайна по λ (б.п.):")
    print(t[[c for c in t.columns if c.startswith("lam_")]].mean().round(2).to_string())
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, "tuning.csv"), index=False, encoding="utf-8-sig")


def main():
    if "--tune" in sys.argv:
        return tune()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    h = C.load_history()
    days = sorted(h.TRADEDATE.unique())
    mkt = market_regimes(h, days)
    mkt.to_csv(os.path.join(OUT, "market.csv"), encoding="utf-8-sig")
    print(f"Режимы рынка: {mkt.regime.value_counts().to_dict()}  "
          f"(порог волатильности стресса {mkt.attrs['vol_hi']:.1f} б.п./день)", flush=True)
    _init()
    rows, curves_rows, err_rows, loo_rows, drop_rows, scen_rows, noise_rows = ([] for _ in range(7))
    with Pool(processes=WORKERS, initializer=_init) as pool:
        for n, r in enumerate(pool.imap(process_day, range(len(DAYS)), chunksize=2), 1):
            if r is not None:
                rows.append(r[0]); curves_rows += r[1]; err_rows += r[2]; loo_rows += r[3]
                drop_rows += r[4]; scen_rows += r[5]; noise_rows += r[6]
            if n % 25 == 0 or n == len(DAYS):
                print(f"  {n}/{len(DAYS)} дней, {time.time() - t0:.0f} c", flush=True)
    for name, rr in [("daily_fit", rows), ("curves", curves_rows), ("bond_errors", err_rows),
                     ("loo_errors", loo_rows), ("sparsity", drop_rows), ("scenarios", scen_rows),
                     ("noise", noise_rows)]:
        pd.DataFrame(rr).to_csv(os.path.join(OUT, name + ".csv"), index=False, encoding="utf-8-sig")
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин. Файлы в {OUT}")


if __name__ == "__main__":
    sys.exit(main())
