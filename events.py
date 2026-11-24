"""
События и ликвидность: прогноз решений Банка России по кривой и настоящий спред бид/аск.

1. Прогноз «к следующему заседанию».
   Накануне каждого заседания (закрытие дня D-1, решение выходит в 13:30 дня D) берём форвардную ставку
   кривой на период действия нового решения: от даты вступления (понедельник после заседания) до даты
   вступления следующего. Форвард — это ставка, которую рынок «закладывает» на этот период.
   Кривая ОФЗ лежит ниже денежного рынка (премия за надёжность/ликвидность), поэтому сырой форвард
   смещён. Поправку берём только из ПРОШЛЫХ заседаний (средняя ошибка на них) — так прогноз честный,
   его можно было сделать в тот же день. Сравниваем с «ставку не изменят» (тест Диболда–Мариано)
   и считаем долю угаданных направлений (снижение / пауза / повышение).
   «Сюрприз ЦБ» = факт − прогноз. Дальше — как сюрпризы двигают кривую в день решения.

2. Настоящий спред бид/аск.
   В файлах КБД (data/raw/zcyc) есть лучшие котировки покупки и продажи по каждой ОФЗ на закрытие
   (≈18:40–18:50). Сравниваем их с оценкой Corwin–Schultz, которую использовал тест на шум цен,
   смотрим, как спред расширяется в стресс, и какая доля ошибок кривых вне выборки укладывается в спред.

Нужны results/*.csv из run_analysis.py.  Запуск:  python events.py  -> results/meetings.csv,
results/spreads_real.csv, results/events.json, results/fig_meetings.png, results/fig_spreads.png
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curves as C
import run_analysis as R

OUT = R.OUT
METH = ["bootstrap", "ns", "ns_fix", "svensson", "spline", "kbd"]
LABEL = {"bootstrap": "Бутстрэп", "ns": "Нельсон–Сигель", "ns_fix": "NS, τ фикс.", "svensson": "Свенссон",
         "spline": "Сплайн", "kbd": "КБД Мосбиржи"}
MIN_PAST = 3        # поправку на премию начинаем считать после 3 прошлых заседаний
EFFECT_LAG = 3      # решение в пятницу вступает в силу в понедельник
STEP = 0.25         # направление: |изменение| < 0,25 п.п. считаем паузой


def simple_from_cont(f):
    """Непрерывная форвардная ставка (доли) -> овернайт-ставка, % годовых (как ключевая)."""
    return 100 * 365 * (np.exp(f / 365) - 1)


def direction(x):
    return np.where(x >= STEP, 1, np.where(x <= -STEP, -1, 0))


def dm_hln(e1, e2):
    """Диболд–Мариано для квадратичных потерь, h = 1, поправка HLN. Отрицательная статистика — e1 лучше."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    from scipy import stats
    s = d.mean() / np.sqrt(d.var(ddof=1) / n)
    s *= np.sqrt((n - 1) / n)  # HLN при h = 1
    return float(s), float(2 * stats.t.sf(abs(s), n - 1))


# ---------------------------------------------------------------- 1. заседания
SHORT_MIN = 0.05    # для прогноза берём и короткие бумаги (от ~2,5 недель): без них короткий конец кривой —
                    # это продолжение линии от бумаг от 3 мес., и форвард на 6 недель показывает «среднюю ставку на год»
FIXED = "ns"        # метод, по которому считаем «сюрприз» (выбран заранее — основной рекомендуемый метод)


def meeting_table():
    R._init()
    days = R.DAYS
    kr = C.load_cbr_rate("keyrate")
    meets = [m for m in C.load_cbr_meetings() if days[0] < m <= days[-1]]
    rows = []
    for j, m in enumerate(meets):
        if m not in days:
            continue
        i = days.index(m)
        d0 = days[i - 1]                                   # закрытие накануне решения
        nxt = meets[j + 1] if j + 1 < len(meets) else m + pd.Timedelta(days=49)
        eff, eff2 = m + pd.Timedelta(days=EFFECT_LAG), nxt + pd.Timedelta(days=EFFECT_LAG)
        k_old, k_new = float(kr[m - pd.Timedelta(days=1)]), float(kr[eff])
        row = {"meeting": m, "prev_day": d0, "k_old": k_old, "k_new": k_new, "change": k_new - k_old,
               "regime": R.MKT.loc[m].regime if m in R.MKT.index else ""}
        for tag, day in (("pre", d0), ("post", m)):
            settle = days[days.index(day) + 1] if days.index(day) + 1 < len(days) else day + pd.offsets.BDay(1)
            hd = R.H[R.H.TRADEDATE == day]
            short = C.market_on(hd, R.CFS, settle, min_years=SHORT_MIN, issue_dates=R.ISSUE)
            base = C.market_on(hd, R.CFS, settle, issue_dates=R.ISSUE)
            if tag == "pre":
                row["n_short"] = sum(b.maturity < 0.25 for b in short)
            fs = C.fit_all(short, methods=[x for x in METH if x != "kbd"])   # для прогноза
            fb = C.fit_all(base, methods=[x for x in METH if x != "kbd"])    # для реакции кривой (как в основном расчёте)
            k = C.load_kbd(day)
            if k:
                fs["kbd"] = fb["kbd"] = R._KBD(k[0])
            t1, t2 = max((eff - settle).days, 1) / 365, (eff2 - settle).days / 365
            for mt in METH:
                if mt in fs:
                    row[f"{tag}_{mt}"] = simple_from_cont(R._fwd(fs[mt], t1, t2))
                if mt in fb:
                    for T in (1, 5, 10):
                        row[f"{tag}_z{T}_{mt}"] = C.to_annual(float(np.ravel(fb[mt].zero(T))[0]))
        rows.append(row)
        print(f"  {m.date()}: {k_old:.2f} -> {k_new:.2f}, NS накануне {row.get('pre_ns', np.nan):.2f}")
    t = pd.DataFrame(rows)
    t["fc_nochange"] = t.k_old
    t["fc_momentum"] = t.k_old + t.change.shift(1).fillna(0)   # «ЦБ сделает тот же шаг, что в прошлый раз»
    n = len(t)
    for mt in METH:
        raw = t[f"pre_{mt}"]
        # (а) в реальном времени: сдвиг на среднюю ошибку сырого форварда по ПРОШЛЫМ заседаниям
        gap = (raw - t.k_new).expanding().mean().shift(1)
        gap[t.index < MIN_PAST] = np.nan
        t[f"fc_{mt}"] = raw - gap
        # (б) регрессия изменения ставки на (форвард − текущая ставка), без самого заседания (leave-one-out)
        x, y, p = (raw - t.k_old).values, t.change.values, np.full(n, np.nan)
        for i in range(n):
            msk = np.arange(n) != i
            a, b = np.linalg.lstsq(np.c_[np.ones(n - 1), x[msk]], y[msk], rcond=None)[0]
            p[i] = a + b * x[i]
        t[f"fcreg_{mt}"] = t.k_old + p
    return t


def _score(ok, col):
    e = ok.k_new - ok[col]
    return e, {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
               "hit": float(np.mean(direction(ok[col] - ok.k_old) == direction(ok.change)))}


def meeting_stats(t):
    ok = t[t.index >= MIN_PAST]     # одинаковый набор заседаний для всех прогнозов
    e0, s0 = _score(ok, "fc_nochange")
    em, sm = _score(ok, "fc_momentum")
    res = {"n_meetings": int(len(t)), "n_eval": int(len(ok)), "n_moves": int((ok.change.abs() >= STEP).sum()),
           "first_eval": str(ok.meeting.iloc[0].date()),
           "n_short_mean": float(t.n_short.mean()), "share_no_short": float((t.n_short == 0).mean()),
           "nochange": s0, "momentum": sm, "methods": {}}
    s, p = dm_hln(em, e0)
    res["momentum"].update(dm_vs_nochange=s, p_vs_nochange=p)
    for mt in METH:
        d = {"raw_bias": float((t[f"pre_{mt}"] - t.k_new).mean()),
             "raw_bias_sd": float((t[f"pre_{mt}"] - t.k_new).std())}
        for tag, col in (("realtime", f"fc_{mt}"), ("regression", f"fcreg_{mt}")):
            e, sc = _score(ok, col)
            sc["dm_vs_nochange"], sc["p_vs_nochange"] = dm_hln(e, e0)
            sc["dm_vs_momentum"], sc["p_vs_momentum"] = dm_hln(e, em)
            d[tag] = sc
        res["methods"][mt] = d
    # сколько вариантов перепробовано: 6 методов × 2 калибровки = 12 -> поправка Бонферрони
    res["n_variants"] = 2 * len(METH)
    # сюрприз = факт − прогноз заранее выбранного метода в реальном времени; реакция кривой в день решения
    ok = ok.assign(surprise=ok.k_new - ok[f"fc_{FIXED}"])
    reac = {}
    for T in (1, 5, 10):
        for mt in ("ns", "spline", "bootstrap", "kbd"):
            dz = (ok[f"post_z{T}_{mt}"] - ok[f"pre_z{T}_{mt}"]) * 100
            X = np.c_[np.ones(len(ok)), ok.surprise]
            b, *_ = np.linalg.lstsq(X, dz, rcond=None)
            resid = dz - X @ b
            r2 = 1 - resid.var() / dz.var() if dz.var() > 0 else np.nan
            reac[f"{mt}_{T}"] = {"slope_bp_per_pp": float(b[1]), "r2": float(r2),
                                 "mean_abs_move_bp": float(dz.abs().mean())}
    res["reaction"] = reac
    res["surprises"] = [{"date": str(r.meeting.date()), "k_old": r.k_old, "k_new": r.k_new,
                         "forecast": float(r[f"fc_{FIXED}"]), "surprise": float(r.surprise)}
                        for _, r in ok.iterrows()]
    return res, ok


def fig_meetings(t, best):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.step(t.meeting, t.k_new, where="post", color="#1f2a44", lw=2, label="Ключевая ставка (факт)")
    ax.plot(t.meeting, t.pre_ns, "o", mfc="none", color="#a0a0a0", ms=6,
            label="Сырой форвард NS накануне (без поправки)")
    ax.plot(t.meeting, t[f"fc_{best}"], "o", color="#c8553d", ms=7,
            label=f"Прогноз: форвард {LABEL[best]} с поправкой по прошлым заседаниям")
    ax.plot(t.meeting, t.fc_momentum, "x", color="#5b8e7d", ms=7, label="«Тот же шаг, что в прошлый раз»")
    ax.set_ylabel("% годовых")
    ax.set_title("Что ждал рынок накануне каждого заседания ЦБ и что решил ЦБ")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_meetings.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2. настоящий спред
def real_spreads():
    rows = []
    for d in R.DAYS:
        p = os.path.join(C.ROOT, "zcyc", pd.Timestamp(d).strftime("%Y-%m-%d") + ".json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p, encoding="utf-8"))
        s = j.get("securities", {})
        if not s.get("data"):
            continue
        df = pd.DataFrame(s["data"], columns=s["columns"])
        df = df[(df.bidprice > 0) & (df.askprice > df.bidprice)]
        for r in df.itertuples():
            mid = (r.askprice + r.bidprice) / 2
            dur = r.crtduration / 365 if r.crtduration else np.nan
            sp_p = (r.askprice - r.bidprice) / mid * 1e4
            # доходности в файле округлены до 0,01% — спред в доходности считаем из цены через дюрацию
            rows.append({"date": d, "secid": r.secid, "spread_price_bp": sp_p, "dur": dur,
                         "spread_yield_bp": sp_p / dur if dur and dur > 0.05 else np.nan})
    sp = pd.DataFrame(rows)
    sp["date"] = pd.to_datetime(sp.date)
    cs = R.H[["TRADEDATE", "SECID", "HALF_SPREAD"]].rename(columns={"TRADEDATE": "date", "SECID": "secid"})
    sp = sp.merge(cs, on=["date", "secid"], how="left")
    sp["cs_spread_price_bp"] = sp.HALF_SPREAD * 2 * 1e4
    sp = sp.merge(R.MKT[["regime"]], left_on="date", right_index=True, how="left")
    sp.drop(columns="HALF_SPREAD").to_csv(os.path.join(OUT, "spreads_real.csv"), index=False, encoding="utf-8-sig")
    return sp


def spread_stats(sp):
    ok = sp.dropna(subset=["cs_spread_price_bp"])
    daily = sp.groupby("date").agg(real=("spread_price_bp", "median"), real_y=("spread_yield_bp", "median"),
                                   regime=("regime", "first"))
    daily_cs = ok.groupby("date").cs_spread_price_bp.median()
    res = {"n_obs": int(len(sp)), "n_days": int(sp.date.nunique()),
           "real_median_price_bp": float(sp.spread_price_bp.median()),
           "real_median_yield_bp": float(sp.spread_yield_bp.median()),
           "cs_median_price_bp": float(ok.cs_spread_price_bp.median()),
           "corr_bond_day": float(np.corrcoef(ok.spread_price_bp, ok.cs_spread_price_bp)[0, 1]),
           "corr_daily_median": float(daily.real.corr(daily_cs)),
           "by_regime": {}, "by_maturity": {}}
    for g, x in sp.groupby("regime"):
        res["by_regime"][g] = {"price_bp": float(x.spread_price_bp.median()),
                               "yield_bp": float(x.spread_yield_bp.median()),
                               "cs_price_bp": float(x.cs_spread_price_bp.median())}
    bins = [0, 2, 5, 8, 40]   # по дюрации, лет
    for (lo, hi), x in zip(zip(bins[:-1], bins[1:]),
                           [sp[(sp.dur >= a) & (sp.dur < b)] for a, b in zip(bins[:-1], bins[1:])]):
        res["by_maturity"][f"{lo}-{hi}"] = {"price_bp": float(x.spread_price_bp.median()),
                                           "yield_bp": float(x.spread_yield_bp.median())}
    # доля ошибок вне выборки, которые меньше половины настоящего спреда (в доходности)
    loo = pd.read_csv(os.path.join(OUT, "loo_errors.csv"), parse_dates=["date"], encoding="utf-8-sig")
    m = loo.merge(sp[["date", "secid", "spread_yield_bp", "regime"]], on=["date", "secid"], how="inner")
    m = m[m.spread_yield_bp > 0]
    inside = {}
    for (mt, rg), x in m.groupby(["method", "regime"]):
        inside.setdefault(mt, {})[rg] = float(np.mean(np.abs(x.err_bp) <= x.spread_yield_bp / 2))
    for mt, x in m.groupby("method"):
        inside[mt]["Все дни"] = float(np.mean(np.abs(x.err_bp) <= x.spread_yield_bp / 2))
    res["loo_inside_half_spread"] = inside
    return res, daily


def fig_spreads(daily):
    fig, ax = plt.subplots(figsize=(10, 4))
    col = {"Спокойно": "#5b8e7d", "Обычно": "#a0a0a0", "Стресс": "#c8553d"}
    for g, x in daily.groupby("regime"):
        ax.scatter(x.index, x.real_y, s=10, color=col.get(g, "#888"), label=g)
    ax.plot(daily.real_y.rolling(20, min_periods=5).median(), color="#1f2a44", lw=1.5, label="Скользящая медиана")
    ax.set_ylabel("б.п. доходности")
    ax.set_title("Настоящий спред бид/аск ОФЗ на закрытие (медиана по выпускам за день)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_spreads.png"), dpi=150)
    plt.close(fig)


def main():
    t = meeting_table()
    t.to_csv(os.path.join(OUT, "meetings.csv"), index=False, encoding="utf-8-sig")
    ms, _ = meeting_stats(t)
    fig_meetings(t, FIXED)
    sp = real_spreads()
    ss, daily = spread_stats(sp)
    fig_spreads(daily)
    out = {"meetings": ms, "spreads": ss}
    json.dump(out, open(os.path.join(OUT, "events.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
