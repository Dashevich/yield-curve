"""
Сводные таблицы и графики по результатам run_analysis.py.
Запуск:  python report.py   ->  results/summary.md + results/fig_*.png + results/numbers.json
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import curves as C
from run_analysis import GRID, OUT, PERIODS, REGIMES, SYNTH, OWN, METHODS

NAMES = {"bootstrap": "Бутстрэп (все выпуски)", "bootstrap_1y": "Бутстрэп (1 выпуск/год)",
         "ns": "Нельсон–Сигель", "svensson": "Свенссон", "ns_fix": "NS с фикс. τ",
         "ns_anchor": "NS + якорь RUONIA", "spline": "Сглаживающий сплайн", "kbd": "КБД Мосбиржи"}
# категориальная палитра (dataviz reference, фиксированный порядок слотов); КБД — нейтральный серый эталон
COLORS = {"ns": "#2a78d6", "bootstrap": "#eb6834", "spline": "#1baf7a", "svensson": "#eda100",
          "bootstrap_1y": "#e87ba4", "ns_fix": "#008300", "ns_anchor": "#6250d6", "kbd": "#7A8B99"}
ORDER = METHODS
SMOOTH = ["ns", "svensson", "ns_fix", "ns_anchor", "spline"]
NUM = {}  # ключевые числа для PDF


def period_of(d):
    for name, a, b in PERIODS:
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            return name
    return None


def rd(name):
    df = pd.read_csv(os.path.join(OUT, name + ".csv"), encoding="utf-8-sig", low_memory=False)
    df["date"] = pd.to_datetime(df.date)
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


def per_group(df, value, methods, by="period", agg="mean"):
    t = df.pivot_table(index=by, columns="method", values=value, aggfunc=agg)
    order = [p[0] for p in PERIODS] if by == "period" else REGIMES
    t = t.reindex([o for o in order if o in t.index])[[m for m in methods if m in t.columns]]
    t.columns = [NAMES[m] for m in t.columns]
    t.index.name = "Период" if by == "period" else "Режим"
    return t


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


# ---------------------------------------------------------------- статистика
def block_boot_rmse(daily_sse, daily_n, reps=2000, block=10, seed=0):
    """90% доверительный интервал RMSE: блочный бутстрэп по дням (блоки по `block` дней подряд),
    чтобы учесть, что соседние дни похожи."""
    rng = np.random.default_rng(seed)
    T = len(daily_sse)
    if T < 3:
        return np.nan, np.nan
    nb = int(np.ceil(T / block))
    out = []
    for _ in range(reps):
        starts = rng.integers(0, max(T - block + 1, 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block, T)) for s in starts])[:T]
        out.append(np.sqrt(daily_sse[idx].sum() / daily_n[idx].sum()))
    return np.percentile(out, 5), np.percentile(out, 95)


def nw_var(x, lag):
    """Дисперсия среднего по Ньюи–Уэсту (учёт автокорреляции)."""
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    v = np.dot(x, x) / T
    for k in range(1, min(lag, T - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * np.dot(x[k:], x[:-k]) / T
    return v / T


def dm_test(e1, e2, h):
    """Тест Диболда–Мариано (с поправкой Harvey–Leybourne–Newbold): одинакова ли точность двух
    прогнозов. d = e1² − e2²; < 0 — первый точнее. h — горизонт в наблюдениях (перекрытие окон)."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    T = len(d)
    v = nw_var(d, h - 1)
    if v <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(v)
    hln = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-6))
    stat = dm * hln
    return stat, 2 * stats.t.sf(abs(stat), T - 1)


def ols_hac(y, X, lag=10, cluster=None):
    """МНК с устойчивыми ошибками: Ньюи–Уэст (временной ряд) или кластеры (панель по дням)."""
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    if cluster is not None:
        S = np.zeros((X.shape[1], X.shape[1]))
        for g in np.unique(cluster):
            s = X[cluster == g].T @ u[cluster == g]
            S += np.outer(s, s)
    else:
        Xu = X * u[:, None]
        S = Xu.T @ Xu
        for k in range(1, lag + 1):
            G = Xu[k:].T @ Xu[:-k]
            S += (1 - k / (lag + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.diag(V)), 1 - u.var() / y.var()


# ---------------------------------------------------------------- отчёт
def main():
    daily, curves, errs, loo = (rd(n) for n in ("daily_fit", "curves", "bond_errors", "loo_errors"))
    drop, scen, noise = (rd(n) for n in ("sparsity", "scenarios", "noise"))
    mkt = pd.read_csv(os.path.join(OUT, "market.csv"), parse_dates=["date"], encoding="utf-8-sig").set_index("date")
    reg = mkt.regime
    for df in (curves, errs, loo):
        df["regime"] = df.date.map(reg)
    n_reg = daily.regime.value_counts().reindex(REGIMES).fillna(0).astype(int)
    v_hi = mkt.vol20_bp.quantile(0.8)
    NUM.update(days=len(daily), n_min=int(daily.n_bonds.min()), n_max=int(daily.n_bonds.max()),
               n_med=float(daily.n_bonds.median()), max_mat=float(daily.max_mat.max()),
               regime_days=n_reg.to_dict(), vol_hi=float(v_hi), tau_fixed=C.TAU_FIXED, spline_lam=C.Spline.LAM)
    tp = os.path.join(OUT, "tuning.csv")
    if os.path.exists(tp):
        tu = pd.read_csv(tp, encoding="utf-8-sig")
        NUM["tuning"] = {"tau_median": float(tu.tau.median()), "tau_q25": float(tu.tau.quantile(.25)),
                         "tau_q75": float(tu.tau.quantile(.75)), "n_days": len(tu),
                         "lam_rmse": {c[4:]: float(tu[c].mean()) for c in tu.columns if c.startswith("lam_")}}
    hh = C.load_history()
    hh = hh[(hh.NUMTRADES >= 5) & hh.WAPRICE.notna()]
    badw = hh.apply(C.bad_waprice, axis=1)
    NUM["bad_waprice"] = {"rows": int(badw.sum()), "of": len(hh), "days": int(hh[badw].TRADEDATE.nunique())}

    out = ["# Результаты: восстановление кривой ОФЗ-ПД, 2024-01 — 2026-10", "",
           f"Торговых дней: {len(daily)}; выпусков в день: {daily.n_bonds.min()}–{daily.n_bonds.max()} "
           f"(медиана {daily.n_bonds.median():.0f}); самый длинный выпуск: до {daily.max_mat.max():.1f} лет.",
           "Ставки — годовые эффективные, %; ошибки — в доходности, б.п.", "",
           "Методы: бутстрэп (2 варианта), Нельсон–Сигель, Свенссон, NS с фиксированным τ "
           f"(τ = {C.TAU_FIXED} г.), NS с якорем короткого конца на RUONIA (±{C.ANCHOR_BAND * 100:.1f} п.п.), "
           "сглаживающий сплайн форвардов (VRP), эталон — КБД Мосбиржи.", ""]

    # ---- 0. Режимы
    rg_by = daily.groupby("regime").agg(n=("n_bonds", "median"), vol=("volume_bn", "median"),
                                        hs=("half_spread_bp", "median")).reindex(REGIMES)
    mk_by = mkt.groupby("regime").agg(v=("vol20_bp", "median"), dd=("drawdown", "median")).reindex(REGIMES)
    t = pd.DataFrame({"Дней": n_reg, "Волатильность 10 лет, б.п./день": mk_by.v,
                      "Просадка RGBI, %": mk_by.dd * 100, "Выпусков в расчёте": rg_by.n,
                      "Оборот, млрд ₽": rg_by.vol, "Полуспред, б.п. доходности": rg_by.hs})
    t.index.name = "Режим"
    NUM["regime_table"] = t.round(2).reset_index().to_dict("records")
    out += ["## 0. Рыночные режимы (по данным, а не по календарю)", "",
            f"Стресс — день, когда скользящая (20 дней) волатильность дневных изменений 10-летней ставки КБД "
            f"выше, чем в 80% дней (≥ {v_hi:.1f} б.п./день), или RGBI ниже своего максимума за квартал больше чем на 7%. "
            "Спокойно — волатильность ниже медианы и просадка меньше 3%. Остальное — «обычно».", "",
            md(t), ""]
    st = daily[daily.regime == "Стресс"].date
    if len(st):
        # сплошные эпизоды стресса
        dd = pd.Series(sorted(st))
        grp = (dd.diff().dt.days > 7).cumsum()
        eps = dd.groupby(grp).agg(["min", "max", "count"])
        eps = eps[eps["count"] >= 3]
        NUM["stress_episodes"] = [(a.strftime("%d.%m.%Y"), b.strftime("%d.%m.%Y"), int(c)) for a, b, c in eps.values]
        out += ["Эпизоды стресса (≥ 3 дней): " + "; ".join(f"{a:%d.%m.%Y}–{b:%d.%m.%Y} ({c} дн.)" for a, b, c in eps.values), ""]

    # ---- 1. Точность
    long = daily.melt(id_vars=["date", "period", "regime"], value_vars=[f"rmse_{m}" for m in ORDER if f"rmse_{m}" in daily],
                      var_name="method", value_name="rmse")
    long["method"] = long.method.str.replace("rmse_", "")
    t1a = per_group(long, "rmse", ORDER)
    out += ["## 1. Точность подгонки", "", "### 1a. RMSE в выборке (средний за день), б.п.", "",
            "Бутстрэп по всем выпускам проходит через каждую цену, поэтому его ошибка в выборке ≈ 0 — "
            "сравнивать его честно можно только вне выборки (1b).", "", md(t1a), ""]
    NUM["in_sample"] = t1a.round(1).to_dict()

    sse = loo.assign(sq=loo.err_bp ** 2).groupby(["method", "date"]).agg(sse=("sq", "sum"), n=("sq", "size")).reset_index()
    sse["regime"] = sse.date.map(reg)
    sse["period"] = sse.date.map(period_of)
    lt = loo.groupby(["period", "method"]).err_bp.apply(rmse).reset_index()
    t1b = per_group(lt, "err_bp", OWN)
    out += ["### 1b. RMSE вне выборки (leave-one-out, каждый день), б.п.", "",
            "Каждую облигацию оцениваем по кривой, построенной без неё. Теперь — на всех днях, а не на каждом 10-м; "
            "параметрические модели стартуют не только с решения на полной выборке, но и из стандартных точек.", "",
            md(t1b), ""]
    NUM["loo_period"] = t1b.round(1).to_dict()

    # по режимам + доверительные интервалы
    rows = {}
    for r in REGIMES:
        rows[r] = {}
        for m in OWN:
            s = sse[(sse.method == m) & (sse.regime == r)].sort_values("date")
            if s.empty:
                continue
            v = np.sqrt(s.sse.sum() / s.n.sum())
            lo, hi = block_boot_rmse(s.sse.values, s.n.values)
            rows[r][NAMES[m]] = f"{v:.1f} [{lo:.1f}; {hi:.1f}]"
            NUM.setdefault("loo_regime", {}).setdefault(m, {})[r] = [round(v, 1), round(lo, 1), round(hi, 1)]
    t = pd.DataFrame(rows).T
    t.index.name = "Режим"
    out += ["### 1b'. RMSE вне выборки по рыночным режимам, б.п. [90% интервал]", "",
            "Интервал — блочный бутстрэп по дням (блоки по 10 дней). Если интервалы двух методов "
            "сильно пересекаются, разница между ними может быть случайной.", "", md(t, "{}"), ""]

    # парные сравнения: разница RMSE и её интервал
    pairs = [("ns", "bootstrap"), ("spline", "ns"), ("ns_fix", "ns"), ("ns_anchor", "ns"), ("svensson", "ns")]
    rows = []
    for a, b in pairs:
        for r in REGIMES:
            sa = sse[(sse.method == a) & (sse.regime == r)].set_index("date")
            sb = sse[(sse.method == b) & (sse.regime == r)].set_index("date")
            j = sa.index.intersection(sb.index)
            if len(j) < 5:
                continue
            sa, sb = sa.loc[j].sort_index(), sb.loc[j].sort_index()
            diff = np.sqrt(sa.sse.sum() / sa.n.sum()) - np.sqrt(sb.sse.sum() / sb.n.sum())
            rng = np.random.default_rng(1)
            T = len(j); nb = int(np.ceil(T / 10)); bs = []
            for _ in range(2000):
                st_ = rng.integers(0, max(T - 9, 1), nb)
                idx = np.concatenate([np.arange(s, min(s + 10, T)) for s in st_])[:T]
                bs.append(np.sqrt(sa.sse.values[idx].sum() / sa.n.values[idx].sum())
                          - np.sqrt(sb.sse.values[idx].sum() / sb.n.values[idx].sum()))
            lo, hi = np.percentile(bs, [5, 95])
            verdict = "лучше" if hi < 0 else ("хуже" if lo > 0 else "неразличимо")
            rows.append({"Сравнение": f"{NAMES[a]} − {NAMES[b]}", "Режим": r, "Разница, б.п.": diff,
                         "90% интервал": f"[{lo:.1f}; {hi:.1f}]", "Вывод": verdict})
            NUM.setdefault("pairs", []).append([a, b, r, round(diff, 1), round(lo, 1), round(hi, 1), verdict])
    t = pd.DataFrame(rows).set_index("Сравнение")
    out += ["### 1b''. Значима ли разница? (минус — первый метод точнее)", "", md(t), ""]

    # короткий конец подробно (NS): откуда большая ошибка до 3 лет
    ln = loo[loo.method == "ns"].copy()
    ln["rank"] = ln.groupby("date").maturity.rank()
    sb = ln[ln.maturity < 3].groupby(pd.cut(ln.maturity, [0, 0.5, 1, 2, 3]), observed=True).err_bp.apply(rmse)
    NUM["short_end"] = {"buckets": {str(k): round(v, 1) for k, v in sb.items()},
                        "shortest": rmse(ln[ln["rank"] == 1].err_bp), "rest_lt3": rmse(ln[(ln.maturity < 3) & (ln["rank"] > 1)].err_bp)}
    NUM["weekday"] = {k: round(v, 1) for k, v in ln.groupby(ln.date.dt.dayofweek).err_bp.apply(rmse).items()}
    # разрыв между мгновенной ставкой NS и RUONIA
    pr = daily.set_index("date").ns_params.dropna().str.split().apply(lambda x: float(x[0]) + float(x[1]))
    ru = C.simple_to_cont(mkt.ruonia.reindex(pr.index))
    gap = (pr - ru) * 1e4
    NUM["ruonia_gap"] = {"median": float(gap.median()), "q25": float(gap.quantile(.25)), "q75": float(gap.quantile(.75))}
    # сколько наблюдений затрагивал баг «купон в день расчётов»
    hh2 = C.load_history(); cfs2 = C.load_cashflows(hh2.SECID.unique()); dl = sorted(hh2.TRADEDATE.unique())
    nxt = {d: dl[i + 1] for i, d in enumerate(dl[:-1])}
    e_ns = errs[errs.method == "ns"]
    flag = [nxt.get(d) is not None and (cfs2[s].date == nxt[d]).any() for d, s in zip(e_ns.date, e_ns.secid)]
    NUM["coupon_on_settle"] = {"rows": int(np.sum(flag)), "of": len(e_ns),
                               "share_tuesday": float(np.mean(e_ns.date[np.array(flag)].dt.dayofweek == 1)) if np.any(flag) else 0}

    loo["bucket"] = pd.cut(loo.maturity, [0, 3, 7, 12, 30], labels=["до 3 лет", "3–7 лет", "7–12 лет", "больше 12 лет"])
    bt = loo.groupby(["bucket", "method"], observed=True).err_bp.apply(rmse).unstack()
    bt = bt[[m for m in OWN if m in bt]]; bt.columns = [NAMES[m] for m in bt.columns]; bt.index.name = "Срок"
    out += ["### 1c. RMSE вне выборки по срокам, б.п.", "", md(bt), ""]
    NUM["loo_bucket"] = bt.round(1).to_dict()

    # регрессия ошибки на рыночные индикаторы
    X = mkt[["vol20_bp", "drawdown", "turnover_bn", "n_traded"]].copy()
    X["drawdown"] = -X.drawdown * 100
    X["turnover_bn"] = np.log(X.turnover_bn.clip(lower=0.1))
    labels = ["Волатильность 10 лет", "Просадка RGBI", "log оборота", "Число торгуемых выпусков"]
    rows = []
    for m in ["bootstrap", "ns", "spline", "svensson"]:
        s = sse[sse.method == m].set_index("date")
        y = np.sqrt(s.sse / s.n)
        d = X.join(y.rename("y"), how="inner").dropna()
        Z = (d[X.columns] - d[X.columns].mean()) / d[X.columns].std()
        b, se, r2 = ols_hac(d.y.values, Z.values, lag=10)
        rows.append({"Метод": NAMES[m], **{lab: f"{b[k + 1]:+.1f} ({b[k + 1] / se[k + 1]:+.1f})" for k, lab in enumerate(labels)},
                     "R²": f"{r2:.2f}"})
        NUM.setdefault("regression", {})[m] = {lab: [round(b[k + 1], 2), round(b[k + 1] / se[k + 1], 2)] for k, lab in enumerate(labels)} | {"r2": round(r2, 3)}
    t = pd.DataFrame(rows).set_index("Метод")
    out += ["### 1d. От чего зависит дневная ошибка вне выборки (регрессия)", "",
            "Коэффициент — на сколько б.п. растёт дневная RMSE вне выборки, если показатель вырастает на одно "
            "стандартное отклонение (остальные при этом фиксированы). В скобках — t-статистика (Ньюи–Уэст, 10 лагов): "
            "|t| > 2 — связь значима.", "", md(t, "{}"), ""]

    # ---- 2. Длинный конец
    c = curves.sort_values("date")
    kb = c[c.method == "kbd"].set_index("date")
    rows = []
    for m in ORDER:
        s = c[c.method == m].set_index("date")
        if s.empty:
            continue
        rows.append({"Метод": NAMES[m], "Ставка 10 лет, средн.": s["z10"].mean(), "Ставка 20 лет, средн.": s["z20"].mean(),
                     "|20 лет − КБД|, б.п.": (s["z20"] - kb["z20"]).abs().mean() * 100,
                     "σ дневн. изм. 10 лет, б.п.": s["z10"].diff().std() * 100,
                     "σ дневн. изм. 20 лет, б.п.": s["z20"].diff().std() * 100,
                     "Форвард 15→20 лет, мин.": s["fwd_15_20"].min(), "Форвард 15→20 лет, макс.": s["fwd_15_20"].max()})
    t = pd.DataFrame(rows).set_index("Метод")
    NUM["long_end"] = t.round(1).to_dict("index")
    out += ["## 2. Длинный конец", "", md(t), ""]
    bad = ((daily.bootstrap_fwd_min < 0) | (daily.bootstrap_fwd_max > 30)).mean()
    NUM["boot_bad_share"] = float(bad)
    out += [f"Наивный бутстрэп: в {bad:.0%} дней мгновенный форвард выходит за пределы [0%, 30%] "
            f"(медиана минимума {daily.bootstrap_fwd_min.median():.1f}%, максимума {daily.bootstrap_fwd_max.median():.1f}%). "
            f"В среднем бутстрэп пропускает {daily.bootstrap_skipped.mean():.1f} выпуска в день (близкие сроки или нет решения).", ""]
    bnd = {m: (daily[f"{m}_at_bound"].astype(str) == "True").mean() for m in ("ns", "svensson", "ns_fix", "ns_anchor") if f"{m}_at_bound" in daily}
    sh = {m: (daily[f"{m}_short_active"].astype(str) == "True").mean() for m in ("ns", "svensson", "ns_fix", "ns_anchor") if f"{m}_short_active" in daily}
    NUM["at_bound"] = bnd; NUM["short_active"] = sh
    out += ["Доля дней, когда параметр упирается в границу / короткая ставка на краю полосы: " +
            "; ".join(f"{NAMES[m]} — {bnd[m]:.0%} / {sh[m]:.0%}" for m in bnd), ""]

    # ---- эффект купона
    out += ["### Эффект купона", ""]
    for m in ["svensson", "ns"]:
        sv = errs[errs.method == m].dropna(subset=["coupon_rate"])
        lowc = sv.coupon_rate < 0.08
        out += [f"Ошибки модели «{NAMES[m]}»: выпуски с купоном < 8% — в среднем {sv[lowc].err_bp.mean():+.1f} б.п., "
                f"остальные — {sv[~lowc].err_bp.mean():+.1f} б.п.", ""]
        NUM.setdefault("coupon_means", {})[m] = [float(sv[lowc].err_bp.mean()), float(sv[~lowc].err_bp.mean())]
    sv = errs[(errs.method == "ns")].dropna(subset=["coupon_rate", "age", "volume"]).copy()
    sv = sv[sv.volume > 0]
    Xc = pd.DataFrame({"coupon": sv.coupon_rate * 100, "age": sv.age, "logvol": np.log(sv.volume),
                       "m37": sv.maturity.between(3, 7).astype(float), "m712": sv.maturity.between(7, 12).astype(float),
                       "m12": (sv.maturity > 12).astype(float)})
    b, se, r2 = ols_hac(sv.err_bp.values, Xc.values, cluster=sv.date.values)
    labs = ["Купон, за 1 п.п.", "Возраст выпуска, за 1 год", "log оборота", "Срок 3–7 лет", "Срок 7–12 лет", "Срок > 12 лет"]
    t = pd.DataFrame({"Коэффициент, б.п.": b[1:], "t-стат.": b[1:] / se[1:]}, index=labs)
    t.index.name = "Фактор"
    NUM["coupon_reg"] = {lab: [round(b[k + 1], 2), round(b[k + 1] / se[k + 1], 2)] for k, lab in enumerate(labs)} | {"r2": round(r2, 3), "n": len(sv)}
    out += ["Регрессия ошибки NS (б.п.; плюс = модель даёт доходность выше рыночной, т.е. рынок платит за бумагу больше модели) "
            "на купон с контролем возраста выпуска, ликвидности и срока. Ошибки кластеризованы по дням.", "",
            md(t, "{:.2f}"), "", f"R² = {r2:.2f}, наблюдений {len(sv)}.", ""]

    # ---- 3. Разреженность
    out += ["## 3. Устойчивость при нехватке выпусков", "",
            "### 3a. Случайно убираем 30% / 50% выпусков", "",
            "10 повторов на день, каждый 5-й день и все стрессовые дни. Сдвиг — RMSE разницы кривых на сроках 1–15 лет "
            "относительно кривой по всем выпускам; ошибка — RMSE на убранных выпусках. Медианы.", ""]
    for share in sorted(drop.share.unique()):
        d = drop[drop.share == share]
        t1 = per_group(d, "curve_shift_bp", OWN, "regime", "median")
        t2 = per_group(d, "oos_rmse_bp", OWN, "regime", "median")
        out += [f"Убрано {share:.0%}: сдвиг кривой, б.п.", "", md(t1), "",
                f"Убрано {share:.0%}: ошибка на убранных выпусках, б.п.", "", md(t2), ""]
        NUM.setdefault("drop", {})[f"{share:.1f}"] = t1.round(1).to_dict()
    out += ["### 3b. Сценарии «как в стресс»", "",
            "В стресс пропадают не случайные бумаги, а длинные и неликвидные. Сдвиг кривой 1–15 лет и ошибка "
            "на убранных выпусках (медианы по дням), а для «без длинных» — ещё и сдвиг 20-летней ставки.", ""]
    for sc_name in scen.scenario.unique():
        d = scen[scen.scenario == sc_name]
        g = d.groupby("method")
        t = pd.DataFrame({"Сдвиг 1–15 лет, б.п.": g.curve_shift_bp.median(), "Ошибка на убранных, б.п.": g.oos_rmse_bp.median(),
                          "|Сдвиг 20 лет|, б.п.": g.shift_20y_bp.apply(lambda x: x.abs().median())}).reindex(OWN).dropna(how="all")
        t.index = [NAMES[m] for m in t.index]; t.index.name = "Метод"
        out += [f"**{sc_name}**", "", md(t), ""]
        NUM.setdefault("scenarios", {})[sc_name] = t.round(1).to_dict("index")

    # ---- 4. Шум цен
    t = per_group(noise, "curve_shift_bp", OWN, "regime", "median")
    t20 = noise.groupby(["regime", "method"]).shift_20y_bp.apply(lambda x: x.abs().median()).unstack()
    t20 = t20.reindex(REGIMES)[[m for m in OWN if m in t20]]; t20.columns = [NAMES[m] for m in t20.columns]; t20.index.name = "Режим"
    NUM["noise"] = t.round(1).to_dict(); NUM["noise20"] = t20.round(1).to_dict()
    out += ["## 4. Шум цен в пределах спреда бид/аск", "",
            "Котировок бид/аск в данных нет — спред оценён по дневным максимуму и минимуму (метод Corwin–Schultz). "
            "Каждую цену случайно сдвигаем в пределах этого спреда (20 повторов) и строим кривые заново. "
            "Медианный сдвиг кривой 1–15 лет, б.п.:", "", md(t), "", "Медианный |сдвиг| 20-летней ставки, б.п.:", "", md(t20), ""]

    # ---- 5. Модельный риск
    rows = {}
    for T in SYNTH:
        p = curves.pivot_table(index="date", columns="method", values=f"px{T}")
        rng_all = (p[[m for m in SMOOTH + ["kbd"] if m in p]].max(axis=1) - p[[m for m in SMOOTH + ["kbd"] if m in p]].min(axis=1))
        by = rng_all.groupby(p.index.map(reg)).median().reindex(REGIMES)
        rows[f"{T} лет"] = by
    t = pd.DataFrame(rows).T
    t.index.name = "Облигация"
    NUM["model_risk"] = t.round(1).to_dict()
    pb = curves.pivot_table(index="date", columns="method", values="px20")
    NUM["model_risk_boot20"] = float((pb.bootstrap - pb.kbd).abs().median())
    out += ["## 5. Модельный риск: одна и та же облигация — разные цены", "",
            "Цена условной облигации с купоном 12% (на 1000 ₽ номинала) по разным гладким кривым и КБД. "
            "Показан разброс (максимум − минимум) между методами, медиана по дням режима, ₽:", "", md(t), "",
            f"Для сравнения: бутстрэп оценивает 20-летнюю облигацию в среднем на {NUM['model_risk_boot20']:.0f} ₽ иначе, чем КБД.", ""]

    # ---- 6. Заседания ЦБ
    meet = [d for d in C.load_cbr_meetings() if daily.date.min() <= d <= daily.date.max()]
    days_all = sorted(daily.date)
    ev_days = []
    for d in meet:  # решение в 13:30 — реакция в тот же торговый день (если торгов нет — в следующий)
        j = np.searchsorted(days_all, d)
        if j < len(days_all):
            ev_days.append(days_all[j])
    kr = C.load_cbr_rate("keyrate")
    dz = {m: c[c.method == m].set_index("date").sort_index()[["z1", "z10"]].diff() * 100 for m in ORDER}
    loo_d = {m: np.sqrt(sse[sse.method == m].set_index("date").sse / sse[sse.method == m].set_index("date").n).sort_index() for m in OWN}
    rows, ratio = [], {m: [] for m in OWN}
    for d in ev_days:
        before = kr.asof(d - pd.Timedelta(days=1)) if kr is not None else np.nan
        after = kr.asof(d + pd.Timedelta(days=4)) if kr is not None else np.nan
        r = {"Дата": f"{d:%d.%m.%Y}", "Ставка": f"{before:g} → {after:g}" if pd.notna(before) else "—",
             "Δ КБД 1 год, б.п.": dz["kbd"].z1.get(d, np.nan), "Δ КБД 10 лет, б.п.": dz["kbd"].z10.get(d, np.nan)}
        for m in ["bootstrap", "ns", "spline"]:
            s = loo_d[m]
            prev = s[(s.index < d) & (s.index >= d - pd.Timedelta(days=30))]
            r[f"Ошибка вне выборки {NAMES[m]}, ×"] = s.get(d, np.nan) / prev.median() if len(prev) else np.nan
        for m in OWN:
            s = loo_d[m]
            prev = s[(s.index < d) & (s.index >= d - pd.Timedelta(days=30))]
            if d in s.index and len(prev):
                ratio[m].append(s[d] / prev.median())
        rows.append(r)
    t = pd.DataFrame(rows).set_index("Дата") if rows else pd.DataFrame()
    # «шум метода сверх рынка»: σ(Δz10 метода − Δz10 КБД) в дни заседаний и в остальные дни
    ex = {}
    evs = set(ev_days)
    for m in OWN:
        e = (dz[m].z10 - dz["kbd"].z10).dropna()
        ex[NAMES[m]] = {"В дни заседаний": e[e.index.isin(evs)].std(), "В остальные дни": e[~e.index.isin(evs)].std(),
                        "Ошибка вне выборки в день заседания, × к обычной": np.median(ratio[m]) if ratio[m] else np.nan}
    te = pd.DataFrame(ex).T
    te.index.name = "Метод"
    NUM["events"] = te.round(2).to_dict("index"); NUM["n_events"] = len(ev_days)
    out += ["## 6. Дни заседаний Банка России", "",
            f"{len(ev_days)} заседаний в выборке (даты — из календаря на cbr.ru). «×» — во сколько раз ошибка вне выборки в день "
            "заседания больше её медианы за предыдущие 30 дней.", "", md(t, "{:.2f}") if len(t) else "—", "",
            "Насколько 10-летняя ставка метода «дёргается» сильнее, чем КБД (σ разницы дневных изменений, б.п.):", "",
            md(te, "{:.2f}"), ""]

    # ---- 7. PCA
    cols = ["z1", "z2", "z3", "z5", "z7", "z10", "z12", "z15"]
    rows = {}
    for m in ORDER:
        s = c[c.method == m].set_index("date").sort_index()[cols].diff().dropna() * 100
        if len(s) < 20:
            continue
        ev = np.linalg.eigvalsh(np.cov(s.values.T))[::-1]
        sh = ev / ev.sum()
        rows[NAMES[m]] = {"1-я компонента (уровень), %": sh[0] * 100, "2-я (наклон), %": sh[1] * 100,
                          "3-я (кривизна), %": sh[2] * 100, "Первые три вместе, %": sh[:3].sum() * 100}
    t = pd.DataFrame(rows).T
    t.index.name = "Метод"
    NUM["pca"] = t.round(1).to_dict("index")
    out += ["## 7. Из чего складываются дневные движения кривой (метод главных компонент)", "",
            "У настоящей кривой почти всё движение — это сдвиг уровня, наклона и изгиба. Если на первые три "
            "компоненты приходится заметно меньше, чем у КБД, — метод добавляет собственный шум.", "", md(t), ""]

    # ---- 8. Прогноз
    out += forecasts(curves, kb)
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(out))
    json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    plots(daily, curves, loo, sse, drop, scen, noise, mkt, ev_days)
    print("\n".join(out))


FC = None


def _realized(series, d0, days=91):
    """Фактическая ставка на окне [d0, d0+days): ежедневное начисление по ставке ЦБ / RUONIA, годовая эффективная, %."""
    w = series[(series.index >= d0) & (series.index < d0 + pd.Timedelta(days=days))]
    if len(w) < days - 3:
        return np.nan
    g = np.prod(1 + w.values / 100 / 365)
    return 100 * (g ** (365 / len(w)) - 1)


def forecasts(curves, kb):
    global FC
    kb3 = kb["z0.25"]
    kr, ru = C.load_cbr_rate("keyrate"), C.load_cbr_rate("ruonia")
    rows = []
    for m in ORDER:
        s = curves[curves.method == m].set_index("date")["fwd_1y_3m"].sort_index()
        for d0, f in s.items():
            tgt = d0 + pd.Timedelta(days=365)
            r = {"date": d0, "method": m, "forecast": f, "kbd_now": kb3.get(d0, np.nan)}
            j = kb3.index.searchsorted(tgt)
            r["fact_kbd"] = kb3.iloc[j] if j < len(kb3) and (kb3.index[j] - tgt).days <= 7 else np.nan
            r["fact_key"] = _realized(kr, tgt) if kr is not None else np.nan
            r["fact_ruonia"] = _realized(ru, tgt) if ru is not None else np.nan
            r["key_now"] = _realized(kr, d0 - pd.Timedelta(days=91)) if kr is not None else np.nan  # «не изменится»
            r["ruonia_now"] = _realized(ru, d0 - pd.Timedelta(days=91)) if ru is not None else np.nan
            rows.append(r)
    fc = pd.DataFrame(rows)
    FC = fc
    out = ["## 8. Форвардная ставка как прогноз", "",
           "Прогноз: форвардная 3-месячная ставка через год. Три варианта «факта»: 3-месячная ставка КБД через год "
           "(как раньше); ставка ЦБ, фактически начисленная за те 3 месяца; то же по RUONIA. «Не изменится» — "
           "сегодняшнее значение того же показателя. Тест Диболда–Мариано: p < 0,05 — разница в точности значима. "
           "Окна прогнозов перекрываются, поэтому независимых наблюдений очень мало (последний столбец).", ""]
    for fact, now, lab in [("fact_kbd", "kbd_now", "3-мес. ставка КБД"), ("fact_key", "key_now", "ключевая ставка ЦБ"),
                           ("fact_ruonia", "ruonia_now", "RUONIA")]:
        rows = []
        for m in ORDER:
            s = fc[(fc.method == m)].dropna(subset=[fact, now]).sort_values("date")
            if len(s) < 30:
                continue
            e, e0 = s.forecast - s[fact], s[now] - s[fact]
            h = 313  # ≈ 15 месяцев в торговых днях: окна «прогноз → конец факта» перекрываются
            dm, p = dm_test(e.values, e0.values, min(h, len(s) - 1))
            rows.append({"Метод": NAMES[m], "Смещение, п.п.": e.mean(), "RMSE, п.п.": rmse(e), "RMSE «не изменится», п.п.": rmse(e0),
                         "DM-стат.": dm, "p-значение": p, "Период прогнозов": f"{s.date.min():%m.%Y}–{s.date.max():%m.%Y}",
                         "Независимых наблюдений ≈": len(s) / 250})
            NUM.setdefault("forecast", {}).setdefault(lab, {})[m] = [round(e.mean(), 2), round(rmse(e), 2), round(rmse(e0), 2),
                                                                     round(dm, 2) if pd.notna(dm) else None,
                                                                     round(p, 3) if pd.notna(p) else None, round(len(s) / 250, 1)]
        if rows:
            t = pd.DataFrame(rows).set_index("Метод")
            out += [f"**Факт — {lab}**", "", md(t, "{:.2f}"), ""]
    return out


# ---------------------------------------------------------------- графики
def _wrap(name):
    import textwrap
    return "\n".join(textwrap.wrap(name, 12))


def _style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9a9893", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e4df", "grid.linewidth": 0.6,
                         "lines.linewidth": 1.6, "legend.frameon": False, "figure.facecolor": "#ffffff"})


def _shade_stress(ax, mkt):
    s = (mkt.regime == "Стресс").astype(int)
    on = None
    for d, v in s.items():
        if v and on is None:
            on = d
        if not v and on is not None:
            ax.axvspan(on, d, color="#eb6834", alpha=0.10, lw=0); on = None
    if on is not None:
        ax.axvspan(on, s.index[-1], color="#eb6834", alpha=0.10, lw=0)


def plots(daily, curves, loo, sse, drop, scen, noise, mkt, ev_days):
    _style()
    h = C.load_history(); cfs = C.load_cashflows(h.SECID.unique())
    days = sorted(h.TRADEDATE.unique())

    # Кривые на ключевые даты
    dates = ["2024-10-30", "2026-07-15", "2026-10-30"]
    titles = ["30.10.2024 — инверсия, ставка 21%", "15.07.2026 — стресс", "30.10.2026 — последняя дата"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    tt = np.linspace(0.1, 20, 300)
    for ax, d, ti in zip(axes, dates, titles):
        d = pd.Timestamp(d)
        if d not in days:
            d = days[np.searchsorted(days, d) - 1]
        i = days.index(d)
        bonds = C.market_on(h[h.TRADEDATE == d], cfs, days[i + 1] if i + 1 < len(days) else d + pd.offsets.BDay(1))
        ru = mkt.ruonia.get(d, np.nan)
        fits = C.fit_all(bonds, short_rate=float(C.simple_to_cont(ru)) if pd.notna(ru) else None,
                         methods=["bootstrap", "ns", "svensson", "spline"])
        par, _ = C.load_kbd(d)
        ax.scatter([b.dur for b in bonds], [C.to_annual(b.ytm) for b in bonds], s=16, color="#9a9893", zorder=3,
                   label="YTM выпусков (по дюрации)")
        for m, f in fits.items():
            ax.plot(tt, C.to_annual(f.zero(tt)), color=COLORS[m], label=NAMES[m])
        ax.plot(tt, C.to_annual(C.kbd_zero(par, tt)), color=COLORS["kbd"], lw=2, ls="--", label=NAMES["kbd"])
        ax.set_title(ti); ax.set_xlabel("Срок, лет"); ax.set_ylim(5, 25)
    axes[0].set_ylabel("Бескупонная ставка, % годовых"); axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_curves_key_dates.png"), dpi=150); plt.close(fig)

    # КБД во времени
    k = curves[curves.method == "kbd"].sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for col, lab, colr in [("z0.25", "3 мес", "#eb6834"), ("z1", "1 год", "#eda100"), ("z5", "5 лет", "#1baf7a"),
                           ("z10", "10 лет", "#2a78d6"), ("z20", "20 лет", "#6250d6")]:
        ax.plot(k.date, k[col], label=lab, color=colr)
    kr = C.load_cbr_rate("keyrate")
    if kr is not None:
        kk = kr[(kr.index >= k.date.min()) & (kr.index <= k.date.max())]
        ax.step(kk.index, kk.values, where="post", color="#0b0b0b", lw=1.2, label="ключевая ставка ЦБ")
    _shade_stress(ax, mkt)
    ax.set_ylabel("% годовых"); ax.set_title("КБД Мосбиржи по срокам и ключевая ставка (оранжевый фон — стресс по данным)")
    ax.legend(ncol=6, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_kbd_history.png"), dpi=150); plt.close(fig)

    # Режимы рынка
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    axes[0].plot(mkt.index, mkt.rgbi, color="#2a78d6"); axes[0].set_ylabel("RGBI")
    axes[0].set_title("Индекс гособлигаций RGBI")
    axes[1].plot(mkt.index, mkt.vol20_bp, color="#2a78d6")
    axes[1].axhline(mkt.vol20_bp.quantile(0.8), color="#52514e", ls="--", lw=1)
    axes[1].text(mkt.index[5], mkt.vol20_bp.quantile(0.8) * 1.04, "порог стресса (80% дней)", fontsize=8, color="#52514e")
    axes[1].set_ylabel("б.п./день"); axes[1].set_title("Волатильность 10-летней ставки КБД (20 дней)")
    for m in ["bootstrap", "ns", "spline"]:
        s = sse[sse.method == m].set_index("date").sort_index()
        axes[2].plot(s.index, np.sqrt(s.sse.rolling(10).sum() / s.n.rolling(10).sum()), color=COLORS[m], label=NAMES[m])
    axes[2].set_ylabel("б.п."); axes[2].set_title("Ошибка вне выборки (скользящая за 10 дней)"); axes[2].legend(fontsize=8, ncol=3)
    for ax in axes:
        _shade_stress(ax, mkt)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_regimes.png"), dpi=150); plt.close(fig)

    # LOO RMSE по месяцам
    loo["month"] = loo.date.dt.to_period("M").dt.to_timestamp()
    mth = loo.groupby(["month", "method"]).err_bp.apply(rmse).unstack()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for meth in ["bootstrap", "bootstrap_1y", "ns", "spline"]:
        ax.plot(mth.index, mth[meth], marker="o", ms=3, color=COLORS[meth], label=NAMES[meth])
    _shade_stress(ax, mkt)
    ax.set_yscale("log"); ax.set_ylabel("RMSE вне выборки, б.п. (лог)"); ax.legend(fontsize=8)
    ax.set_title("Точность вне выборки (leave-one-out) по месяцам")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_loo_rmse.png"), dpi=150); plt.close(fig)

    # Форварды бутстрэпа на последнюю дату
    d = days[-1]
    bonds = C.market_on(h[h.TRADEDATE == d], cfs, d + pd.offsets.BDay(1))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for meth, b in [("bootstrap", C.Bootstrap(bonds)), ("bootstrap_1y", C.Bootstrap(bonds, bucket=1.0))]:
        ax.step(b.knots[1:], C.to_annual(b.fwd), where="pre", color=COLORS[meth], label=NAMES[meth])
    t15 = tt[tt < 15]
    for meth, f in [("svensson", C.Parametric(bonds, "sv")), ("spline", C.Spline(bonds))]:
        ax.plot(t15, C.to_annual(np.ravel(f.forward(t15))), color=COLORS[meth], label=NAMES[meth])
    ax.axhline(0, color="#ccc", lw=0.8); ax.set_xlabel("Срок, лет"); ax.set_ylabel("Мгновенный форвард, %")
    ax.set_title(f"Форвардные ставки, {pd.Timestamp(d):%d.%m.%Y}: бутстрэп «пилит» из-за эффекта купона")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_bootstrap_forwards.png"), dpi=150); plt.close(fig)

    # Разреженность: случайная + сценарии
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    meths = ["bootstrap", "bootstrap_1y", "ns", "svensson", "ns_fix", "spline"]
    x = np.arange(len(meths)); w = 0.38
    for j, share in enumerate(sorted(drop.share.unique())):
        v = [drop[(drop.share == share) & (drop.method == mm)].curve_shift_bp.median() for mm in meths]
        axes[0].bar(x + (j - 0.5) * w, v, w - 0.02, label=f"убрано {share:.0%}", color=["#a9c8ee", "#2a78d6"][j])
    axes[0].set_xticks(x, [_wrap(NAMES[mm]) for mm in meths], fontsize=7.5)
    axes[0].set_ylabel("Сдвиг кривой 1–15 лет, б.п. (медиана)"); axes[0].set_title("Случайная нехватка выпусков"); axes[0].legend()
    scs = list(scen.scenario.unique())
    w = 0.8 / max(len(scs), 1)
    for j, sc in enumerate(scs):
        v = [scen[(scen.scenario == sc) & (scen.method == mm)].oos_rmse_bp.median() for mm in meths]
        axes[1].bar(x + (j - (len(scs) - 1) / 2) * w, v, w - 0.02, label=sc, color=["#eb6834", "#1baf7a", "#6250d6"][j % 3])
    axes[1].set_xticks(x, [_wrap(NAMES[mm]) for mm in meths], fontsize=7.5)
    axes[1].set_ylabel("Ошибка на убранных выпусках, б.п. (медиана)"); axes[1].set_title("Сценарии «как в стресс»")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sparsity.png"), dpi=150); plt.close(fig)

    # Шум цен по режимам
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.8 / 3
    for j, r in enumerate(REGIMES):
        v = [noise[(noise.regime == r) & (noise.method == mm)].curve_shift_bp.median() for mm in meths]
        ax.bar(x + (j - 1) * w, v, w - 0.02, label=r, color=["#a9c8ee", "#5b96e0", "#1d4f91"][j])
    ax.set_xticks(x, [_wrap(NAMES[mm]) for mm in meths], fontsize=8)
    ax.set_ylabel("Сдвиг кривой 1–15 лет, б.п. (медиана)"); ax.set_title("Чувствительность к шуму цен в пределах спреда")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_noise.png"), dpi=150); plt.close(fig)

    # Заседания ЦБ: ошибка вне выборки вокруг дня заседания
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m in ["bootstrap", "ns", "spline", "svensson"]:
        s = np.sqrt(sse[sse.method == m].set_index("date").sse / sse[sse.method == m].set_index("date").n).sort_index()
        idx = list(s.index)
        prof = []
        for d in ev_days:
            if d not in s.index:
                continue
            j = idx.index(d)
            base = s.iloc[max(j - 25, 0):max(j - 5, 1)].median()
            prof.append([s.iloc[j + k] / base if 0 <= j + k < len(s) else np.nan for k in range(-5, 6)])
        if prof:
            ax.plot(range(-5, 6), np.nanmedian(prof, axis=0), marker="o", ms=5, color=COLORS[m], label=NAMES[m])
    ax.axvline(0, color="#52514e", ls="--", lw=1); ax.axhline(1, color="#9a9893", lw=0.8)
    ax.set_xlabel("Торговые дни относительно заседания ЦБ"); ax.set_ylabel("Ошибка вне выборки / обычная")
    ax.set_title("Ошибка вне выборки вокруг заседаний Банка России (медиана)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_events.png"), dpi=150); plt.close(fig)

    # Модельный риск: разброс цены 10-летней облигации
    p = curves.pivot_table(index="date", columns="method", values="px10").sort_index()
    sm = [m for m in SMOOTH + ["kbd"] if m in p]
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(p.index, p[sm].max(axis=1) - p[sm].min(axis=1), color="#2a78d6", label="гладкие методы и КБД")
    ax.plot(p.index, (p.bootstrap - p.kbd).abs(), color="#eb6834", lw=1, alpha=0.8, label="|бутстрэп − КБД|")
    _shade_stress(ax, mkt)
    ax.set_ylabel("₽ на 1000 ₽ номинала"); ax.set_yscale("log")
    ax.set_title("Модельный риск: разброс цены 10-летней облигации (купон 12%) между методами"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_model_risk.png"), dpi=150); plt.close(fig)

    # Форвард как прогноз
    if FC is not None and not FC.empty:
        fig, ax = plt.subplots(figsize=(12, 4.2))
        s = FC[FC.method == "kbd"].sort_values("date")
        x = s.date + pd.Timedelta(days=365)
        ax.plot(x, s.forecast, color="#2a78d6", label="прогноз: форвард КБД, сделанный за год")
        ax.plot(x, s.fact_kbd, color="#0b0b0b", lw=2, label="факт: 3-мес. ставка КБД")
        ax.plot(x, s.fact_key, color="#6250d6", lw=1.4, label="факт: ключевая ставка (начисленная за 3 мес.)")
        ax.plot(x, s.kbd_now, color="#eb6834", ls="--", label="«не изменится» (ставка годом ранее)")
        ax.set_ylabel("% годовых"); ax.set_title("Форвардная ставка как прогноз 3-месячной ставки через год"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_forward_forecast.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
