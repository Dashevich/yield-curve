"""
Сводные таблицы и графики по результатам run_analysis.py.
Запуск:  python report.py   ->  results/summary.md + results/fig_*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curves as C
from run_analysis import GRID, OUT, PERIODS

NAMES = {"bootstrap": "Бутстрэп (все выпуски)", "bootstrap_1y": "Бутстрэп (1 выпуск/год)",
         "ns": "Нельсон–Сигель", "svensson": "Свенссон", "kbd": "КБД Мосбиржи"}
COLORS = {"bootstrap": "#B4501F", "bootstrap_1y": "#E07A45", "ns": "#2A6F97",
          "svensson": "#14213D", "kbd": "#7A8B99"}
ORDER = list(NAMES)


def period_of(d):
    for name, a, b in PERIODS:
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            return name
    return None


def rd(name):
    df = pd.read_csv(os.path.join(OUT, name + ".csv"), encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df.date)
    for col in df.columns:  # скаляры, записанные как "[x]"
        if col != "date" and not pd.api.types.is_numeric_dtype(df[col]) and df[col].astype(str).str.startswith("[").any():
            df[col] = pd.to_numeric(df[col].astype(str).str.strip("[]"), errors="coerce")
    df["period"] = df.date.map(period_of)
    return df


def md(df, fmt="{:.1f}"):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        vals = [fmt.format(v) if isinstance(v, (float, np.floating)) and pd.notna(v) else ("—" if pd.isna(v) else str(v))
                for v in r.values]
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(lines)


def per_period(df, value, methods, agg="mean"):
    t = df.pivot_table(index="period", columns="method", values=value, aggfunc=agg)
    t = t.reindex([p[0] for p in PERIODS])[[m for m in methods if m in t.columns]]
    t.columns = [NAMES[m] for m in t.columns]
    t.index.name = "Период"
    return t


def main():
    daily, curves, errs, loo, drop = (rd(n) for n in ("daily_fit", "curves", "bond_errors", "loo_errors", "sparsity"))
    out = ["# Результаты: восстановление кривой ОФЗ-ПД, 2024-01 — 2026-10", "",
           f"Торговых дней: {len(daily)}; выпусков в день: {daily.n_bonds.min()}–{daily.n_bonds.max()} "
           f"(медиана {daily.n_bonds.median():.0f}); самый длинный выпуск: до {daily.max_mat.max():.1f} лет.",
           "Ставки — годовые эффективные, %; ошибки — в доходности, б.п.", "",
           "Калибровка NS/Свенссона с ограничениями (см. curves.Parametric): τ1 ≥ 0,25 г., у Свенссона τ2 ≥ τ1 + 1 г., "
           "|β2|,|β3| ≤ 30 п.п., мгновенная ставка в нуле — в полосе ±200 б.п. от доходности самого короткого выпуска. "
           "Без них Свенссон вырождался (τ1 → 0, всплеск короткой ставки до ~50%; β2 и β3 на границах с разными знаками). "
           "Проверка на каждом 10-м дне: NS упирается в границу в 8% дней, Свенссон — в 66% (короткая ставка на краю полосы — в 32%); "
           "это признак избыточной параметризации Свенссона для ~30 выпусков.", ""]

    # ---- 1. Точность
    long = daily.melt(id_vars=["date", "period"], value_vars=[f"rmse_{m}" for m in ORDER],
                      var_name="method", value_name="rmse")
    long["method"] = long.method.str.replace("rmse_", "")
    out += ["## 1. Точность подгонки", "",
            "### 1a. RMSE в выборке (средний за день), б.п.", "",
            "Бутстрэп по всем выпускам проходит через каждую цену, поэтому его ошибка в выборке ≈ 0 — "
            "сравнивать его честно можно только вне выборки (1b).", "",
            md(per_period(long, "rmse", ORDER)), ""]
    loo["abs"] = loo.err_bp.abs()
    lt = loo.groupby(["period", "method"]).err_bp.apply(lambda e: np.sqrt(np.mean(e ** 2))).reset_index()
    out += ["### 1b. RMSE вне выборки (leave-one-out, каждый 10-й день), б.п.", "",
            "Каждую облигацию оцениваем по кривой, построенной без неё.", "",
            md(per_period(lt, "err_bp", ORDER[:4])), ""]
    loo["bucket"] = pd.cut(loo.maturity, [0, 3, 7, 12, 30], labels=["до 3 лет", "3–7 лет", "7–12 лет", "больше 12 лет"])
    bt = loo.groupby(["bucket", "method"], observed=True).err_bp.apply(lambda e: np.sqrt(np.mean(e ** 2))).unstack()
    bt = bt[ORDER[:4]]; bt.columns = [NAMES[m] for m in bt.columns]; bt.index.name = "Срок"
    out += ["### 1c. RMSE вне выборки по срокам, б.п.", "", md(bt), ""]

    # ---- 2. Длинный конец
    c = curves.sort_values("date")
    out += ["## 2. Длинный конец", ""]
    rows = []
    kb = c[c.method == "kbd"].set_index("date")
    for m in ORDER:
        s = c[c.method == m].set_index("date")
        rows.append({"Метод": NAMES[m],
                     "Ставка 10 лет, средн.": s["z10"].mean(),
                     "Ставка 20 лет, средн.": s["z20"].mean(),
                     "|20 лет − КБД|, б.п.": (s["z20"] - kb["z20"]).abs().mean() * 100,
                     "σ дневн. изм. 10 лет, б.п.": s["z10"].diff().std() * 100,
                     "σ дневн. изм. 20 лет, б.п.": s["z20"].diff().std() * 100,
                     "Форвард 15→20 лет, мин.": s["fwd_15_20"].min(),
                     "Форвард 15→20 лет, макс.": s["fwd_15_20"].max()})
    t = pd.DataFrame(rows).set_index("Метод")
    out += [md(t), ""]
    bad = ((daily.bootstrap_fwd_min < 0) | (daily.bootstrap_fwd_max > 30)).mean()
    out += [f"Наивный бутстрэп: в {bad:.0%} дней мгновенный форвард выходит за пределы [0%, 30%] "
            f"(медиана минимума {daily.bootstrap_fwd_min.median():.1f}%, максимума {daily.bootstrap_fwd_max.median():.1f}%).", ""]

    # ---- эффект купона
    sv = errs[errs.method == "svensson"].dropna(subset=["coupon_rate"])
    lowc = sv.coupon_rate < 0.08
    out += ["### Эффект купона", "",
            f"Ошибки Свенссона: выпуски с купоном < 8% — в среднем {sv[lowc].err_bp.mean():+.1f} б.п., "
            f"остальные — {sv[~lowc].err_bp.mean():+.1f} б.п. (плюс = модель даёт доходность выше рыночной, "
            "т.е. рынок оценивает низкокупонные бумаги дороже). Это и ломает наивный бутстрэп на длинном конце.", ""]

    # ---- 3. Разреженность
    out += ["## 3. Устойчивость при нехватке выпусков", "",
            "Случайно убираем 30% / 50% выпусков (10 повторов на день, каждый 10-й день). "
            "Сдвиг — RMSE разницы кривых на сроках 1–15 лет относительно кривой по всем выпускам; "
            "ошибка — RMSE на убранных выпусках.", ""]
    for share in sorted(drop.share.unique()):
        d = drop[drop.share == share]
        t1 = per_period(d, "curve_shift_bp", ORDER[:4], "median")
        t2 = per_period(d, "oos_rmse_bp", ORDER[:4], "median")
        out += [f"### Убрано {share:.0%}: сдвиг кривой (медиана), б.п.", "", md(t1), "",
                f"### Убрано {share:.0%}: ошибка на убранных выпусках (медиана), б.п.", "", md(t2), ""]

    # ---- 4. Форварды как прогноз
    kb3 = kb["z0.25"]
    fc = []
    for m in ORDER:
        s = c[c.method == m].set_index("date")["fwd_1y_3m"]
        for d0, f in s.items():
            tgt = d0 + pd.Timedelta(days=365)
            j = kb3.index.searchsorted(tgt)
            if j >= len(kb3) or (kb3.index[j] - tgt).days > 7:
                continue
            fc.append({"date": d0, "method": m, "forecast": f, "actual": kb3.iloc[j], "rw": kb3.loc[d0]})
    fc = pd.DataFrame(fc)
    if not fc.empty:
        fc["err"] = fc.forecast - fc.actual
        fc["rw_err"] = fc.rw - fc.actual
        g = fc.groupby("method")
        t = pd.DataFrame({"Смещение, п.п.": g.err.mean(), "RMSE, п.п.": g.err.apply(lambda e: np.sqrt(np.mean(e ** 2))),
                          "RMSE «ставка не изменится», п.п.": g.rw_err.apply(lambda e: np.sqrt(np.mean(e ** 2)))}).reindex(ORDER)
        t.index = [NAMES[m] for m in t.index]; t.index.name = "Метод"
        out += ["## 4. Форвардная ставка как прогноз", "",
                f"Прогноз: форвардная 3-месячная ставка через год (с {fc.date.min():%d.%m.%Y} по {fc.date.max():%d.%m.%Y}). "
                "Факт: 3-месячная ставка КБД через 365 дней. Эталон — «ставка не изменится».", "",
                md(t, "{:.2f}"), "",
                "Смещение > 0 — рынок закладывал более высокие ставки, чем случилось (премия за срок + "
                "недооценённое смягчение).", ""]
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(out))
    plots(daily, curves, loo, drop, fc)
    print("\n".join(out))


def plots(daily, curves, loo, drop, fc):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    h = C.load_history(); cfs = C.load_cashflows(h.SECID.unique())
    days = sorted(h.TRADEDATE.unique())

    # Кривые на ключевые даты
    dates = ["2024-10-30", "2026-07-15", "2026-10-30"]
    titles = ["30.10.2024 — инверсия, ставка 21%", "15.07.2026 — стресс", "30.10.2026 — последняя дата"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    tt = np.linspace(0.1, 20, 300)
    for ax, d, ti in zip(axes, dates, titles):
        d = pd.Timestamp(d); i = days.index(d)
        bonds = C.market_on(h[h.TRADEDATE == d], cfs, days[i + 1] if i + 1 < len(days) else d + pd.offsets.BDay(1))
        fits = C.fit_all(bonds)
        par, _ = C.load_kbd(d)
        ax.scatter([b.dur for b in bonds], [C.to_annual(b.ytm) for b in bonds], s=14, color="#999", zorder=3,
                   label="YTM выпусков (по дюрации)")
        for m, f in fits.items():
            ax.plot(tt, C.to_annual(f.zero(tt)), color=COLORS[m], lw=1.6, label=NAMES[m])
        ax.plot(tt, C.to_annual(C.kbd_zero(par, tt)), color=COLORS["kbd"], lw=2, ls="--", label=NAMES["kbd"])
        ax.set_title(ti); ax.set_xlabel("Срок, лет"); ax.set_ylim(5, 25)
    axes[0].set_ylabel("Бескупонная ставка, % годовых"); axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_curves_key_dates.png"), dpi=150); plt.close(fig)

    # КБД во времени
    k = curves[curves.method == "kbd"].sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for col, lab, colr in [("z0.25", "3 мес", "#E07A45"), ("z1", "1 год", "#B4501F"), ("z5", "5 лет", "#2A6F97"),
                           ("z10", "10 лет", "#14213D"), ("z20", "20 лет", "#7A8B99")]:
        ax.plot(k.date, k[col], label=lab, color=colr, lw=1.4)
    ax.axvspan(pd.Timestamp("2026-06-15"), pd.Timestamp("2026-08-15"), color="#E07A45", alpha=0.12, label="стресс лета 2026")
    ax.set_ylabel("% годовых"); ax.set_title("КБД Мосбиржи: ставки по срокам"); ax.legend(ncol=6, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_kbd_history.png"), dpi=150); plt.close(fig)

    # LOO RMSE по месяцам
    loo["month"] = loo.date.dt.to_period("M").dt.to_timestamp()
    m = loo.groupby(["month", "method"]).err_bp.apply(lambda e: np.sqrt(np.mean(e ** 2))).unstack()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for meth in ["bootstrap", "bootstrap_1y", "ns", "svensson"]:
        ax.plot(m.index, m[meth], marker="o", ms=3, color=COLORS[meth], label=NAMES[meth])
    ax.axvspan(pd.Timestamp("2026-06-15"), pd.Timestamp("2026-08-15"), color="#E07A45", alpha=0.12)
    ax.set_yscale("log"); ax.set_ylabel("RMSE вне выборки, б.п. (лог)"); ax.legend(fontsize=8)
    ax.set_title("Точность вне выборки (leave-one-out) по месяцам")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_loo_rmse.png"), dpi=150); plt.close(fig)

    # Форварды бутстрэпа на последнюю дату
    d = days[-1]
    bonds = C.market_on(h[h.TRADEDATE == d], cfs, d + pd.offsets.BDay(1))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for meth, b in [("bootstrap", C.Bootstrap(bonds)), ("bootstrap_1y", C.Bootstrap(bonds, bucket=1.0))]:
        ax.step(b.knots[1:], C.to_annual(b.fwd), where="pre", color=COLORS[meth], label=NAMES[meth])
    sv = C.Parametric(bonds, "sv")
    ax.plot(tt[tt < 15], C.to_annual(sv.forward(tt[tt < 15])), color=COLORS["svensson"], label=NAMES["svensson"])
    ax.axhline(0, color="#ccc", lw=0.8); ax.set_xlabel("Срок, лет"); ax.set_ylabel("Мгновенный форвард, %")
    ax.set_title(f"Форвардные ставки, {pd.Timestamp(d):%d.%m.%Y}: бутстрэп «пилит» из-за эффекта купона")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_bootstrap_forwards.png"), dpi=150); plt.close(fig)

    # Разреженность
    fig, ax = plt.subplots(figsize=(8, 4.2))
    meths = ["bootstrap", "bootstrap_1y", "ns", "svensson"]
    x = np.arange(len(meths)); w = 0.38
    for j, share in enumerate(sorted(drop.share.unique())):
        v = [drop[(drop.share == share) & (drop.method == mm)].curve_shift_bp.median() for mm in meths]
        ax.bar(x + (j - 0.5) * w, v, w, label=f"убрано {share:.0%}", color=["#C9CFDA", "#14213D"][j])
    ax.set_xticks(x, [NAMES[mm] for mm in meths], fontsize=8); ax.set_ylabel("Сдвиг кривой 1–15 лет, б.п. (медиана)")
    ax.set_title("Устойчивость к нехватке выпусков"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sparsity.png"), dpi=150); plt.close(fig)

    # Форвард как прогноз
    if fc is not None and not fc.empty:
        fig, ax = plt.subplots(figsize=(12, 4.2))
        s = fc[fc.method == "kbd"].sort_values("date")
        ax.plot(s.date + pd.Timedelta(days=365), s.forecast, color="#2A6F97", label="прогноз: форвард КБД, сделанный за год")
        ax.plot(s.date + pd.Timedelta(days=365), s.actual, color="#14213D", lw=2, label="факт: 3-мес. ставка КБД")
        ax.plot(s.date + pd.Timedelta(days=365), s.rw, color="#E07A45", ls="--", label="«не изменится» (ставка годом ранее)")
        ax.set_ylabel("% годовых"); ax.set_title("Форвардная ставка как прогноз 3-месячной ставки через год"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_forward_forecast.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
