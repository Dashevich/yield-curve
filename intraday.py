"""
Внутридневные кривые: точный взгляд на стрессовые дни и дни заседаний ЦБ.

По 10-минутным свечам (download_intraday.py) строим кривые каждые 30 минут: цена бумаги — последняя
сделка до конца окна, если она была не раньше чем за час (иначе бумага в этот момент не участвует).
НКД и дата расчётов — те же, что у дневных итогов этого дня (T+1).

Что считаем:
  1. Заседания ЦБ: изменение ставок с 13:30 (последний снимок до решения) до 14:30 против изменения
     за весь день (закрытие накануне -> закрытие дня решения). Чем лучше внутридневная реакция объясняется
     сюрпризом ЦБ (events.py), тем «чище» измерение.
  2. Внутридневная волатильность 2- и 10-летней ставки в спокойные / стрессовые дни / дни заседаний, по методам.
     Собственный шум метода виден по автокорреляции 30-минутных изменений: у чистого шума она около −0,5,
     у настоящего движения рынка — около 0.
  3. Спред: оценка Corwin–Schultz по соседним 10-минутным свечам и оценка Ролла по изменениям цен —
     против настоящего спреда на закрытие из файлов КБД.

Запуск:  python intraday.py   -> results/intraday_curves.csv, results/intraday.json, results/fig_intraday_*.png
"""

import json
import os
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curves as C
import run_analysis as R

OUT = R.OUT
CANDLES = os.path.join(C.ROOT, "candles")
METH = ["bootstrap", "ns", "spline"]
LABEL = {"bootstrap": "Бутстрэп", "ns": "Нельсон–Сигель", "spline": "Сплайн", "kbd": "КБД"}
STEP_MIN = 30
STALE_MIN = 60
START, END = "07:00", "19:00"
DECISION = "13:30"     # Банк России публикует решение в 13:30 МСК
AFTER = "14:30"
TENORS = (1, 2, 5, 10)
MIN_BONDS = 8

DAYS = H = CFS = None


def _init():
    global DAYS, H, CFS
    DAYS = sorted(C.load_history().TRADEDATE.unique())
    R._init()
    H, CFS = R.H, R.CFS


def load_candles(day):
    d = pd.Timestamp(day).strftime("%Y-%m-%d")
    folder = os.path.join(CANDLES, d)
    out = []
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        c = pd.read_csv(os.path.join(folder, f))
        if c.empty:
            continue
        c["secid"] = f[:-4]
        out.append(c)
    if not out:
        return None
    c = pd.concat(out, ignore_index=True)
    c["begin"] = pd.to_datetime(c["begin"])
    c["end"] = pd.to_datetime(c["end"])
    return c[c.volume > 0]


def _cs(hi, lo):
    """Corwin–Schultz по парам соседних интервалов: полный спред в долях цены (медиана)."""
    hi, lo = np.asarray(hi, float), np.asarray(lo, float)
    if len(hi) < 3:
        return np.nan
    b = np.log(hi[1:] / lo[1:]) ** 2 + np.log(hi[:-1] / lo[:-1]) ** 2
    g = np.log(np.maximum(hi[1:], hi[:-1]) / np.minimum(lo[1:], lo[:-1])) ** 2
    k = 3 - 2 * np.sqrt(2)
    a = (np.sqrt(2 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
    s = np.clip(2 * (np.exp(a) - 1) / (1 + np.exp(a)), 0, None)
    return float(np.median(s))


def _roll(close):
    """Оценка Ролла: спред = 2·√(−cov(Δp_t, Δp_{t−1})), в долях цены; NaN, если ковариация > 0."""
    p = np.log(np.asarray(close, float))
    d = np.diff(p)
    if len(d) < 5:
        return np.nan
    c = np.cov(d[1:], d[:-1])[0, 1]
    return float(2 * np.sqrt(-c)) if c < 0 else np.nan


def process(task):
    day, kind = task
    day = pd.Timestamp(day)
    c = load_candles(day)
    if c is None:
        return [], []
    i = DAYS.index(day)
    settle = DAYS[i + 1] if i + 1 < len(DAYS) else day + pd.offsets.BDay(1)
    hd = H[H.TRADEDATE == day].set_index("SECID")
    # спреды по свечам
    sprows = []
    for s, x in c.groupby("secid"):
        x = x.sort_values("begin")
        sprows.append({"date": day, "kind": kind, "secid": s, "n_candles": len(x),
                       "cs10_bp": _cs(x.high, x.low) * 1e4, "roll_bp": _roll(x.close) * 1e4})
    # снимки кривой
    rows = []
    grid = pd.date_range(day + pd.Timedelta(START + ":00"), day + pd.Timedelta(END + ":00"),
                         freq=f"{STEP_MIN}min")
    for t_end in grid:
        last = c[c.end < t_end].sort_values("end").groupby("secid").tail(1)
        last = last[last.end >= t_end - pd.Timedelta(minutes=STALE_MIN)]
        snap = hd.loc[hd.index.intersection(last.secid)].copy()
        if len(snap) < MIN_BONDS:
            continue
        snap["WAPRICE"] = last.set_index("secid").close.reindex(snap.index)
        snap["NUMTRADES"] = 99
        snap["LOW"] = snap["HIGH"] = np.nan          # дневной диапазон к снимку не относится
        bonds = C.market_on(snap.reset_index(), CFS, settle)
        if len(bonds) < MIN_BONDS:
            continue
        fits = C.fit_all(bonds, methods=METH)
        for m, f in fits.items():
            z = np.ravel(f.zero(np.array(TENORS, float)))
            e = C.yield_errors(bonds, f.zero)
            rows.append({"date": day, "kind": kind, "time": t_end.strftime("%H:%M"), "method": m,
                         "n_bonds": len(bonds), "rmse_bp": float(np.sqrt(np.mean(e ** 2))),
                         **{f"z{T}": C.to_annual(v) for T, v in zip(TENORS, z)}})
    return rows, sprows


# ---------------------------------------------------------------- анализ
def meeting_reaction(cur, daily_curves, surprises):
    """Реакция на решение ЦБ: окно 13:30 -> 14:30 против целого дня."""
    sur = pd.Series({pd.Timestamp(s["date"]): s["surprise"] for s in surprises})
    out, rows = {}, []
    for m in METH:
        x = cur[(cur.kind == "Заседание ЦБ") & (cur.method == m)]
        for d, g in x.groupby("date"):
            g = g.set_index("time")
            if DECISION not in g.index or AFTER not in g.index:
                continue
            dd = daily_curves[daily_curves.method == m].set_index("date")
            j = dd.index.get_indexer([d])[0]
            r = {"date": d, "method": m, "surprise": sur.get(d, np.nan)}
            for T in TENORS:
                if T in (1, 5, 10) and j > 0:
                    r[f"day_{T}"] = (dd[f"z{T}"].iloc[j] - dd[f"z{T}"].iloc[j - 1]) * 100
                r[f"win_{T}"] = (g.loc[AFTER, f"z{T}"] - g.loc[DECISION, f"z{T}"]) * 100
            rows.append(r)
    t = pd.DataFrame(rows)
    for m in METH:
        x = t[(t.method == m)].dropna(subset=["surprise"])
        for T in (1, 5, 10):
            for tag in ("win", "day"):
                y = x[f"{tag}_{T}"].dropna()
                s = x.surprise.loc[y.index]
                if len(y) < 5:
                    continue
                X = np.c_[np.ones(len(y)), s]
                b, *_ = np.linalg.lstsq(X, y, rcond=None)
                r2 = 1 - np.var(y - X @ b) / np.var(y)
                out[f"{m}_{tag}_{T}"] = {"slope": float(b[1]), "r2": float(r2), "n": int(len(y)),
                                         "mean_abs_bp": float(y.abs().mean())}
            # какая доля дневного движения пришлась на час после решения
            xx = x.dropna(subset=[f"day_{T}", f"win_{T}"])
            if len(xx):
                out[f"{m}_share_{T}"] = float(np.sum(np.abs(xx[f"win_{T}"])) / np.sum(np.abs(xx[f"day_{T}"])))
    # во сколько раз движение за полчаса после решения больше обычного получасового движения в те же дни
    for m in METH:
        x = cur[(cur.kind == "Заседание ЦБ") & (cur.method == m)]
        for T in (1, 10):
            jump, usual = [], []
            for d, g in x.groupby("date"):
                g = g.sort_values("time").set_index("time")
                dz = g[f"z{T}"].diff().abs() * 100
                nxt = (pd.to_datetime(DECISION, format="%H:%M") + pd.Timedelta(minutes=STEP_MIN)).strftime("%H:%M")
                if nxt in dz.index:
                    jump.append(dz[nxt])
                    usual.append(dz.drop(nxt).median())
            out[f"{m}_jump_{T}"] = {"jump_bp": float(np.median(jump)), "usual_bp": float(np.median(usual)),
                                    "ratio": float(np.median(jump) / np.median(usual))}
    return out, t


def intraday_vol(cur):
    """Реализованная волатильность за день (√Σ квадратов 30-минутных изменений), б.п., и автокорреляция."""
    out = {}
    for (kind, m), x in cur.groupby(["kind", "method"]):
        vols, ac = {T: [] for T in (2, 10)}, {T: [] for T in (2, 10)}
        for d, g in x.groupby("date"):
            g = g.sort_values("time")
            for T in (2, 10):
                dz = g[f"z{T}"].diff().dropna().values * 100
                if len(dz) >= 6:
                    vols[T].append(np.sqrt(np.sum(dz ** 2)))
                    ac[T].append(np.corrcoef(dz[1:], dz[:-1])[0, 1])
        out.setdefault(kind, {})[m] = {
            **{f"rv{T}_bp": float(np.median(vols[T])) if vols[T] else np.nan for T in (2, 10)},
            **{f"ac{T}": float(np.nanmedian(ac[T])) if ac[T] else np.nan for T in (2, 10)},
            "rmse_bp": float(x.rmse_bp.median()), "n_bonds": float(x.n_bonds.median()),
            "n_days": int(x.date.nunique())}
    return out


def spread_compare(sp):
    real = pd.read_csv(os.path.join(OUT, "spreads_real.csv"), parse_dates=["date"], encoding="utf-8-sig")
    m = sp.merge(real[["date", "secid", "spread_price_bp", "cs_spread_price_bp"]], on=["date", "secid"], how="inner")
    m = m[m.n_candles >= 10]
    out = {"n": int(len(m))}
    for kind, x in list(m.groupby("kind")) + [("Все", m)]:
        out[kind] = {"real_bp": float(x.spread_price_bp.median()),
                     "cs_daily_bp": float(x.cs_spread_price_bp.median()),
                     "cs10_bp": float(x.cs10_bp.median()),
                     "roll_bp": float(x.roll_bp.median()),
                     "roll_defined": float(x.roll_bp.notna().mean()),
                     "corr_cs10": float(x[["spread_price_bp", "cs10_bp"]].corr().iloc[0, 1]),
                     "corr_roll": float(x[["spread_price_bp", "roll_bp"]].corr().iloc[0, 1]),
                     "corr_cs_daily": float(x[["spread_price_bp", "cs_spread_price_bp"]].corr().iloc[0, 1])}
    return out


def fig_meeting_days(cur):
    x = cur[(cur.kind == "Заседание ЦБ") & (cur.method == "ns")]
    days = sorted(x.date.unique())
    if not days:
        return
    n = len(days)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.2 * rows), sharex=True)
    for ax, d in zip(np.ravel(axes), days):
        g = x[x.date == d].sort_values("time")
        tt = pd.to_datetime(g.time, format="%H:%M")
        for T, col in ((1, "#c8553d"), (10, "#1f2a44")):
            ax.plot(tt, (g[f"z{T}"] - g[f"z{T}"].iloc[0]) * 100, color=col, lw=1.3, label=f"{T} год" if T == 1 else f"{T} лет")
        ax.axvline(pd.to_datetime(DECISION, format="%H:%M"), color="#888", ls="--", lw=0.8)
        ax.set_title(pd.Timestamp(d).strftime("%d.%m.%Y"), fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H"))
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    np.ravel(axes)[0].legend(fontsize=7)
    fig.suptitle("Дни заседаний ЦБ: ставки NS внутри дня, б.п. от первого снимка (пунктир — 13:30, решение)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_intraday_meetings.png"), dpi=140)
    plt.close(fig)


def fig_vol(vol):
    kinds = [k for k in ("Спокойно", "Стресс", "Заседание ЦБ") if k in vol]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    col = {"bootstrap": "#c8553d", "ns": "#1f2a44", "spline": "#5b8e7d"}
    w = 0.25
    for ax, T in zip(axes, (2, 10)):
        for j, m in enumerate(METH):
            v = [vol[k][m][f"rv{T}_bp"] for k in kinds]
            ax.bar(np.arange(len(kinds)) + (j - 1) * w, v, w * 0.92, color=col[m], label=LABEL[m])
        ax.set_xticks(np.arange(len(kinds)), kinds)
        ax.set_title(f"Ставка {T} лет: движение внутри дня, б.п.")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_intraday_vol.png"), dpi=150)
    plt.close(fig)


def main():
    _init()
    kinds = pd.read_csv(os.path.join(CANDLES, "days.csv"), parse_dates=["date"]).set_index("date").kind
    tasks = [(d, k) for d, k in kinds.items() if os.path.isdir(os.path.join(CANDLES, d.strftime("%Y-%m-%d")))]
    print(f"{len(tasks)} дней со свечами")
    with Pool(R.WORKERS, initializer=_init) as pool:
        res = pool.map(process, tasks, chunksize=1)
    cur = pd.DataFrame([r for a, _ in res for r in a])
    sp = pd.DataFrame([r for _, b in res for r in b])
    cur.to_csv(os.path.join(OUT, "intraday_curves.csv"), index=False, encoding="utf-8-sig")
    sp.to_csv(os.path.join(OUT, "intraday_spreads.csv"), index=False, encoding="utf-8-sig")
    daily = pd.read_csv(os.path.join(OUT, "curves.csv"), parse_dates=["date"], encoding="utf-8-sig")
    ev = json.load(open(os.path.join(OUT, "events.json"), encoding="utf-8"))
    reac, _ = meeting_reaction(cur, daily, ev["meetings"]["surprises"])
    vol = intraday_vol(cur)
    spc = spread_compare(sp)
    fig_meeting_days(cur)
    fig_vol(vol)
    out = {"n_days": {k: int(v) for k, v in cur.groupby("kind").date.nunique().items()},
           "snapshots_per_day": float(cur[cur.method == "ns"].groupby("date").size().median()),
           "reaction": reac, "vol": vol, "spread": spc}
    json.dump(out, open(os.path.join(OUT, "intraday.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
