"""
Восстановление кривой бескупонных ставок по ОФЗ-ПД.

Модуль содержит:
  * загрузку данных из data/raw (итоги торгов, графики купонов, КБД Мосбиржи,
    индекс RGBI, ключевая ставка / RUONIA / календарь заседаний Банка России);
  * КБД Мосбиржи (G-кривая: Нельсон–Сигель + 9 гауссовых поправок);
  * бутстрэп (кусочно-постоянные форварды между погашениями);
  * Нельсон–Сигель и Свенссон с калибровкой по ошибкам в доходности;
  * Нельсон–Сигель с фиксированным τ (Diebold–Li) и с «якорем» короткого конца на RUONIA;
  * сглаживающий сплайн форвардной кривой (VRP, Waggoner 1997).

Ставки внутри модуля — непрерывно начисляемые, в долях (0.15 = 15%).
Время — в годах ACT/365 от даты расчётов (T+1).
"""

import dataclasses
import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import brentq, least_squares

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")


# ---------------------------------------------------------------- данные
def load_history():
    """Итоги торгов ОФЗ-ПД (SU25*/SU26*, постоянный купон) за все дни."""
    frames = []
    for f in sorted(glob.glob(os.path.join(ROOT, "history", "*.csv"))):
        if os.path.getsize(f) < 10:  # выходной день — пустой файл
            continue
        frames.append(pd.read_csv(f, encoding="utf-8", low_memory=False))
    h = pd.concat(frames, ignore_index=True)
    fixed = set(_fixed_coupon_secids())
    h = h[h.SECID.isin(fixed)].copy()
    h["TRADEDATE"] = pd.to_datetime(h.TRADEDATE)
    h["MATDATE"] = pd.to_datetime(h.MATDATE)
    return h


def _fixed_coupon_secids():
    out = []
    for f in glob.glob(os.path.join(ROOT, "securities", "*.csv")):
        d = pd.read_csv(f, encoding="utf-8").set_index("name")["value"]
        if "постоянным купонным" in str(d.get("ISSUENAME", "")) and str(d.get("FACEUNIT", "SUR")) in ("SUR", "RUB"):
            out.append(d["SECID"])
    return sorted(out)


def load_cashflows(secids):
    """{secid: DataFrame(date, amount)} — купоны + погашение номинала, на 1 облигацию."""
    out = {}
    for s in secids:
        c = pd.read_csv(os.path.join(ROOT, "bondization", f"{s}_coupons.csv"), encoding="utf-8")
        a = pd.read_csv(os.path.join(ROOT, "bondization", f"{s}_amortizations.csv"), encoding="utf-8")
        cf = pd.concat([
            pd.DataFrame({"date": pd.to_datetime(c.coupondate), "amount": c.value.astype(float)}),
            pd.DataFrame({"date": pd.to_datetime(a.amortdate), "amount": a.value.astype(float)}),
        ])
        out[s] = cf.groupby("date", as_index=False).amount.sum().sort_values("date")
    return out


def load_issue_dates(secids):
    """{secid: дата начала первого купонного периода} — приближённо дата размещения."""
    out = {}
    for s in secids:
        c = pd.read_csv(os.path.join(ROOT, "bondization", f"{s}_coupons.csv"), encoding="utf-8")
        out[s] = pd.to_datetime(c.startdate).min()
    return out


def load_kbd(day):
    """Параметры КБД и опубликованные Мосбиржей доходности на сроки (для проверки)."""
    p = os.path.join(ROOT, "zcyc", pd.Timestamp(day).strftime("%Y-%m-%d") + ".json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p, encoding="utf-8"))
    if not j["params"]["data"]:
        return None
    par = dict(zip(j["params"]["columns"], j["params"]["data"][0]))
    yy = pd.DataFrame(j["yearyields"]["data"], columns=j["yearyields"]["columns"])
    return par, yy[["period", "value"]]


def load_rgbi():
    """Индекс RGBI: Series(дата -> закрытие)."""
    p = os.path.join(ROOT, "rgbi.csv")
    if not os.path.exists(p):
        return None
    r = pd.read_csv(p, encoding="utf-8")
    r["TRADEDATE"] = pd.to_datetime(r.TRADEDATE)
    return r.set_index("TRADEDATE").CLOSE.astype(float).sort_index()


def _html_rows(path):
    html = open(path, encoding="utf-8").read()
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = [re.sub(r"<[^>]+>|\s+", " ", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cells) >= 2:
            rows.append(cells)
    return rows


def load_cbr_rate(name):
    """Ключевая ставка ('keyrate') или RUONIA ('ruonia') с сайта ЦБ, % годовых.
    Series по календарным дням (выходные — последнее известное значение)."""
    p = os.path.join(ROOT, name + ".html")
    if not os.path.exists(p):
        return None
    rows = _html_rows(p)
    s = pd.Series({pd.to_datetime(r[0], dayfirst=True): float(r[1].replace(",", ".").replace(" ", ""))
                   for r in rows if re.match(r"\d\d\.\d\d\.\d{4}$", r[0])}).sort_index()
    return s.reindex(pd.date_range(s.index[0], s.index[-1])).ffill()


MONTHS = {m: i + 1 for i, m in enumerate(["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
                                           "августа", "сентября", "октября", "ноября", "декабря"])}


def load_cbr_meetings():
    """Даты заседаний Совета директоров ЦБ по ключевой ставке (из календаря на cbr.ru)."""
    p = os.path.join(ROOT, "calendar.html")
    if not os.path.exists(p):
        return []
    t = open(p, encoding="utf-8").read()
    t = re.sub(r"<[^>]+>", " ", t).replace("&nbsp;", " ").replace("&#160;", " ")
    t = re.sub(r"\s+", " ", t)
    out = set()
    for d, m, y in re.findall(r"(\d{1,2}) ([а-я]+) (20\d\d) года Заседание Совета директоров", t):
        if m in MONTHS:
            out.add(pd.Timestamp(int(y), MONTHS[m], int(d)))
    return sorted(out)


# ---------------------------------------------------------------- КБД Мосбиржи
def _kbd_ab():
    k, a, b = 1.6, [0.0, 0.6], [0.6]
    for i in range(1, 8):
        a.append(a[-1] + a[1] * k ** i)
    for i in range(1, 9):
        b.append(b[-1] * k)
    return np.array(a), np.array(b)


KBD_A, KBD_B = _kbd_ab()


def kbd_zero(par, t):
    """Непрерывная бескупонная ставка КБД (доли) на срок t лет."""
    t = np.maximum(np.asarray(t, float), 1e-6)
    b0, b1, b2, tau = par["B1"], par["B2"], par["B3"], par["T1"]
    x = t / tau
    g = b0 + (b1 + b2) * (1 - np.exp(-x)) / x - b2 * np.exp(-x)
    for i in range(9):
        gi = par[f"G{i + 1}"]
        if gi:
            g = g + gi * np.exp(-(t - KBD_A[i]) ** 2 / KBD_B[i] ** 2)
    return g / 1e4


def to_annual(r):
    """Непрерывная ставка -> годовая эффективная, %."""
    return 100 * (np.exp(r) - 1)


def simple_to_cont(r_pct, days=1):
    """Простая ставка, % годовых, на срок days дней -> непрерывная, доли."""
    return 365 / days * np.log1p(np.asarray(r_pct, float) / 100 * days / 365)


# ---------------------------------------------------------------- рынок одного дня
@dataclass
class Bond:
    secid: str
    t: np.ndarray       # сроки потоков, лет
    cf: np.ndarray      # суммы потоков, руб.
    dirty: float        # грязная цена, руб.
    ytm: float          # непрерывная доходность к погашению (из нашей цены)
    dur: float          # дюрация Маколея при ytm, лет
    maturity: float
    volume: float       # оборот, руб.
    ntrades: int
    coupon: float = np.nan      # ставка купона, доли (из итогов торгов)
    half_spread: float = np.nan  # оценка полуспреда бид/аск, руб. на облигацию
    age: float = np.nan          # лет с размещения


def bond_price(bond, zero_fn):
    return float(np.sum(bond.cf * np.exp(-zero_fn(bond.t) * bond.t)))


def ytm_cont(t, cf, price):
    f = lambda y: np.sum(cf * np.exp(-y * t)) - price
    return brentq(f, -0.5, 2.0)


def _ytm_dur(t, a, dirty):
    y = ytm_cont(t, a, dirty)
    w = a * np.exp(-y * t)
    return y, float(np.sum(w * t) / np.sum(w))


def with_price(b, dirty):
    """Та же облигация с другой грязной ценой (для теста на шум цен)."""
    y, d = _ytm_dur(b.t, b.cf, dirty)
    return dataclasses.replace(b, dirty=dirty, ytm=y, dur=d)


def bad_waprice(r, tol=0.001):
    """Средневзвешенная цена вне [минимум; максимум] дня (с допуском 0,1%) — так в данных ISS бывает
    (~0,2% строк, напр. 26212 и 26236 24.07.2026: WAPRICE 97,5 при диапазоне 90,8–91,4)."""
    return (pd.notna(r.WAPRICE) and pd.notna(r.LOW) and pd.notna(r.HIGH) and r.HIGH > 0
            and (r.WAPRICE > r.HIGH * (1 + tol) or r.WAPRICE < r.LOW * (1 - tol)))


def market_on(h_day, cashflows, settle, min_trades=5, min_years=0.25, issue_dates=None):
    """Список облигаций дня с ценами и потоками после даты расчётов.

    Фильтры: есть цена (средневзвешенная или закрытия), >= min_trades сделок,
    срок до погашения >= min_years (короткие выпуски шумные из-за НКД и ликвидности).
    Если средневзвешенная цена выходит за дневной диапазон — это ошибка данных, берём цену закрытия.
    Потоки — начиная с даты расчётов включительно (см. комментарий ниже).
    """
    bonds = []
    for r in h_day.itertuples():
        price = r.WAPRICE if pd.notna(r.WAPRICE) else r.CLOSE
        if bad_waprice(r):  # средневзвешенная вне дневного диапазона — ошибка данных, берём закрытие
            price = r.CLOSE if pd.notna(r.CLOSE) and r.LOW <= r.CLOSE <= r.HIGH else (r.LOW + r.HIGH) / 2
        if pd.isna(price) or pd.isna(r.ACCINT) or (r.NUMTRADES or 0) < min_trades:
            continue
        cf = cashflows[r.SECID]
        # купон, выплачиваемый в сам день расчётов, достаётся покупателю: сделка вторника (расчёты в среду =
        # дата купона) идёт с полным НКД. С условием «> settle» такие бумаги «дорожали» на купон —
        # доходность расходилась с биржевой на ~130 б.п.; с «>=» совпадает с YIELDATWAP до ~1 б.п.
        cf = cf[cf.date >= settle]
        if cf.empty:
            continue
        t = (cf.date - settle).dt.days.values / 365.0
        if t[-1] < min_years:
            continue
        face = r.FACEVALUE if pd.notna(r.FACEVALUE) else 1000.0
        dirty = price / 100 * face + r.ACCINT
        a = cf.amount.values.astype(float)
        try:
            y, dur = _ytm_dur(t, a, dirty)
        except ValueError:
            continue
        cp = getattr(r, "COUPONPERCENT", np.nan)
        hs = getattr(r, "HALF_SPREAD", np.nan)  # доля цены (см. run_analysis.add_spreads)
        age = (settle - issue_dates[r.SECID]).days / 365 if issue_dates and r.SECID in issue_dates else np.nan
        bonds.append(Bond(r.SECID, t, a, dirty, y, dur, float(t[-1]), float(r.VALUE or 0), int(r.NUMTRADES or 0),
                          coupon=float(cp) / 100 if pd.notna(cp) and cp else np.nan,
                          half_spread=float(hs) * dirty if pd.notna(hs) else np.nan, age=age))
    bonds.sort(key=lambda b: b.maturity)
    return bonds


def yield_errors(bonds, zero_fn):
    """Ошибки модели в доходности, б.п. (модельная YTM минус рыночная)."""
    out = []
    for b in bonds:
        p = bond_price(b, zero_fn)
        out.append((ytm_cont(b.t, b.cf, p) - b.ytm) * 1e4)
    return np.array(out)


class _Flat:
    """Все потоки всех облигаций одним массивом — чтобы цены считались без цикла по облигациям."""

    def __init__(self, bonds):
        self.t = np.concatenate([b.t for b in bonds])
        self.c = np.concatenate([b.cf for b in bonds])
        self.idx = np.repeat(np.arange(len(bonds)), [len(b.t) for b in bonds])
        self.n = len(bonds)
        self.P = np.array([b.dirty for b in bonds])
        self.D = np.array([b.dur for b in bonds])

    def prices_from_integral(self, integ):  # integ = ∫0^t f = z(t)·t в точках self.t
        return np.bincount(self.idx, self.c * np.exp(-integ), minlength=self.n)

    def yield_resid(self, model_prices):
        """Ошибка цены, переведённая в доходность: ΔP / (P·D), б.п."""
        return (model_prices - self.P) / (self.P * self.D) * 1e4


# ---------------------------------------------------------------- бутстрэп
class Bootstrap:
    """Кусочно-постоянные мгновенные форварды между погашениями облигаций.

    Узлы — сроки погашения; z(t)*t линейна между узлами (форвард постоянен).
    До первого узла форвард равен первому, за последним — постоянен (плоская экстраполяция).
    Каждая следующая облигация оценивается точно — подбирается форвард на её участке.
    Выпуски, погашающиеся ближе 30 дней к предыдущему узлу, пропускаются
    (иначе форвард на коротком участке «взрывается» от шума цен); пропущенные — в self.skipped.
    bucket=1.0 — вариант с отбором: по одному самому ликвидному выпуску на каждый год срока.
    """

    def __init__(self, bonds, min_gap=30 / 365, bucket=None):
        if bucket:  # «сглаженный» вариант: самый ликвидный выпуск в каждом интервале длиной bucket лет
            best = {}
            for b in bonds:
                k = int(b.maturity // bucket)
                if k not in best or b.volume > best[k].volume:
                    best[k] = b
            bonds = [best[k] for k in sorted(best)]
            min_gap = max(min_gap, bucket / 2)
        self.knots = [0.0]
        self.fwd = []
        self.used = []
        self.skipped = 0
        for b in bonds:
            if b.maturity - self.knots[-1] < min_gap:
                self.skipped += 1
                continue
            self.knots.append(b.maturity)
            self.fwd.append(0.1)
            f = lambda x: self._set_last(x) or (bond_price(b, self.zero) - b.dirty)
            try:
                self.fwd[-1] = brentq(f, -0.5, 1.5)
            except ValueError:
                self.knots.pop(); self.fwd.pop()
                self.skipped += 1
                continue
            self.used.append(b.secid)
        self.knots = np.array(self.knots)
        self.fwd = np.array(self.fwd)

    def _set_last(self, x):
        self.fwd[-1] = x

    def zero(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        k = np.asarray(self.knots)
        f = np.asarray(self.fwd)
        cum = np.concatenate([[0.0], np.cumsum(f * np.diff(k))])  # ∫f до узла
        idx = np.clip(np.searchsorted(k, t, side="right") - 1, 0, len(f) - 1)
        integ = cum[idx] + f[idx] * (t - k[idx])
        return np.where(t > 1e-9, integ / np.maximum(t, 1e-9), f[0])

    def forward(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        idx = np.clip(np.searchsorted(np.asarray(self.knots), t, side="right") - 1, 0, len(self.fwd) - 1)
        return np.asarray(self.fwd)[idx]


# ---------------------------------------------------------------- Нельсон–Сигель, Свенссон
def ns_zero(p, t):
    b0, b1, b2, tau = p
    t = np.maximum(np.asarray(t, float), 1e-6)
    x = t / tau
    l1 = (1 - np.exp(-x)) / x
    return b0 + b1 * l1 + b2 * (l1 - np.exp(-x))


def sv_zero(p, t):
    b0, b1, b2, b3, tau1, tau2 = p
    t = np.maximum(np.asarray(t, float), 1e-6)
    x1, x2 = t / tau1, t / tau2
    l1 = (1 - np.exp(-x1)) / x1
    l2 = (1 - np.exp(-x2)) / x2
    return b0 + b1 * l1 + b2 * (l1 - np.exp(-x1)) + b3 * (l2 - np.exp(-x2))


def ns_forward(p, t):
    b0, b1, b2, tau = p
    x = np.asarray(t, float) / tau
    return b0 + b1 * np.exp(-x) + b2 * x * np.exp(-x)


def sv_forward(p, t):
    b0, b1, b2, b3, tau1, tau2 = p
    x1, x2 = np.asarray(t, float) / tau1, np.asarray(t, float) / tau2
    return b0 + b1 * np.exp(-x1) + b2 * x1 * np.exp(-x1) + b3 * x2 * np.exp(-x2)


# τ для варианта «NS с фиксированным τ» (Diebold–Li): медиана τ свободного NS по всем дням
# выборки 2024–2026 (пересчитывается скриптом run_analysis.py --tune и вписывается сюда).
TAU_FIXED = 1.28


class Parametric:
    """Калибровка NS/Свенссона: минимизируем ошибки цен, делённые на (цена × дюрация),
    т.е. приближённо ошибки в доходности. Мульти-старт по τ, т.к. задача невыпуклая.

    kind: "ns" — Нельсон–Сигель; "sv" — Свенссон; "nsfix" — NS с фиксированным τ = TAU_FIXED
    (остаются 3 параметра, задача почти линейная и решается однозначно).

    Ограничения (без них Свенссон вырождается: τ1 -> 0 даёт всплеск на коротком конце,
    τ1 ≈ τ2 — коллинеарность, β2 и β3 уходят на границы с разными знаками):
      * β0 (долгосрочный уровень) в [2%, 40%]; |β1| ≤ 50 п.п.; |β2|, |β3| ≤ 30 п.п.;
      * τ1 в [0.25, 10] лет; у Свенссона τ2 = τ1 + Δ, Δ в [1, 30] лет (горбы разнесены);
      * мгновенная ставка в нуле β0+β1 — не дальше short_band от «короткой ставки»: по умолчанию
        это доходность самого короткого выпуска ± 2 п.п.; если задан short_rate (RUONIA) —
        полоса вокруг него (вариант «NS + якорь RUONIA»). Внутри полосы штрафа нет, за её
        пределами — штраф, как за ошибку SHORT_W б.п. на каждый б.п. выхода.
    """

    SHORT_BAND = 0.02
    SHORT_W = 1.0

    def __init__(self, bonds, kind="ns", x0=None, only_x0=False, short_rate=None, short_band=None):
        self.kind = kind
        sv = kind == "sv"
        fix = kind == "nsfix"
        self.fn = sv_zero if sv else ns_zero
        self.fwd_fn = sv_forward if sv else ns_forward
        F = _Flat(bonds)
        y_long = bonds[-1].ytm
        y_short = bonds[0].ytm
        ref = y_short if short_rate is None else short_rate
        band = self.SHORT_BAND if short_band is None else short_band

        def to_params(q):  # внутренние переменные -> (β..., τ1[, τ2])
            if sv:
                return np.r_[q[:5], q[4] + q[5]]
            if fix:
                return np.r_[q, TAU_FIXED]
            return np.asarray(q, float)

        def from_params(p):
            p = np.asarray(p, float)
            if sv:
                return np.r_[p[:5], p[5] - p[4]]
            if fix:
                return p[:3]
            return p

        def resid(q):
            p = to_params(q)
            r = F.yield_resid(F.prices_from_integral(self.fn(p, F.t) * F.t))
            out = max(abs(p[0] + p[1] - ref) - band, 0.0)
            return np.append(r, self.SHORT_W * out * 1e4)

        if sv:
            lo, hi = [0.02, -0.5, -0.3, -0.3, 0.25, 1.0], [0.4, 0.5, 0.3, 0.3, 10, 30]
            starts = [[y_long, ref - y_long, 0.0, 0.0, t1, dt]
                      for t1, dt in ((0.5, 2.5), (1, 4), (1.5, 6), (3, 7), (0.8, 14))]
        elif fix:
            lo, hi = [0.02, -0.5, -0.3], [0.4, 0.5, 0.3]
            starts = [[y_long, ref - y_long, 0.0], [y_long, ref - y_long, 0.05], [y_long, ref - y_long, -0.05]]
        else:
            lo, hi = [0.02, -0.5, -0.3, 0.25], [0.4, 0.5, 0.3, 10]
            starts = [[y_long, ref - y_long, 0.0, tau] for tau in (0.5, 1.5, 3, 6)]
        if x0 is not None:
            w = list(from_params(x0))
            starts = [w] if only_x0 else [w] + starts
        best = None
        lo_a, hi_a = np.array(lo), np.array(hi)
        for s in starts:
            s = np.clip(s, lo_a + 1e-6, hi_a - 1e-6)
            try:
                r = least_squares(resid, s, bounds=(lo, hi), method="trf", max_nfev=400)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        self.params = to_params(best.x)
        self.cost = best.cost
        span = hi_a - lo_a
        self.at_bound = bool(np.any((best.x - lo_a < 1e-3 * span) | (hi_a - best.x < 1e-3 * span)))
        self.short_active = bool(abs(self.params[0] + self.params[1] - ref) > band - 1e-5)

    def zero(self, t):
        return self.fn(self.params, t)

    def forward(self, t):
        return self.fwd_fn(self.params, t)


# ---------------------------------------------------------------- сглаживающий сплайн (VRP)
class Spline:
    """Сглаживающий сплайн мгновенной форвардной кривой (variable roughness penalty, Waggoner 1997;
    так строят кривые Банк Англии и ФРС).

    Форвард f(t) — кубический B-сплайн с узлами KNOTS. Минимизируем
        Σ (ошибка в доходности, б.п.)²  +  ∫ λ(t) · f''(t)² dt,
    где λ(t) растёт со сроком: на коротком конце кривой разрешено изгибаться (там много
    информации — решения ЦБ), на длинном — почти нельзя. За последним выпуском дополнительно
    штрафуется наклон f'(t): форвард стремится к постоянному, а не уходит по прямой.
    Масштаб штрафа LAM подобран по ошибке вне выборки (run_analysis.py --tune).
    """

    KNOTS = [0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 13, 16, 20]
    T_END = 30.0
    LAM = 0.3        # λ на участке 1–10 лет (б.п.² ошибки за (б.п./год²)²·год)
    LAM_SHORT = 0.01  # множитель λ до 1 года
    LAM_LONG = 30.0  # множитель λ после 10 лет
    LAM_FLAT = 1.0   # штраф на наклон форварда за последним выпуском

    def __init__(self, bonds, lam=None, x0=None):
        lam = self.LAM if lam is None else lam
        k = 3
        knots = np.r_[[0.0] * (k + 1), self.KNOTS, [self.T_END] * (k + 1)]
        n = len(knots) - k - 1
        self.basis = BSpline(knots, np.eye(n), k, extrapolate=True)
        self.anti = self.basis.antiderivative()
        self.anti0 = self.anti(0.0)
        F = _Flat(bonds)
        A = self.anti(F.t) - self.anti0                       # ∫0^t B_j, (n_потоков × n)
        g = np.linspace(0, self.T_END, 241)
        dg = g[1] - g[0]
        lam_t = lam * np.where(g < 1, self.LAM_SHORT, np.where(g > 10, self.LAM_LONG, 1.0))
        D2 = self.basis.derivative(2)(g) * np.sqrt(lam_t * dg)[:, None] * 1e4
        beyond = g > bonds[-1].maturity
        D1 = self.basis.derivative(1)(g)[beyond] * np.sqrt(self.LAM_FLAT * dg) * 1e4
        S = np.zeros((F.n, len(F.t)))
        S[F.idx, np.arange(len(F.t))] = 1.0
        scale = 1e4 / (F.P * F.D)

        def resid(c):
            return np.r_[F.yield_resid(F.prices_from_integral(A @ c)), D2 @ c, D1 @ c]

        def jac(c):
            e = F.c * np.exp(-(A @ c))
            jp = -(S @ (e[:, None] * A)) * scale[:, None]
            return np.vstack([jp, D2, D1])

        c0 = np.full(n, np.mean([b.ytm for b in bonds])) if x0 is None else np.asarray(x0, float)
        r = least_squares(resid, c0, jac=jac, method="trf", max_nfev=200)
        self.params = r.x
        self.cost = r.cost

    def forward(self, t):
        return self.basis(np.asarray(t, float)) @ self.params

    def zero(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        integ = (self.anti(t) - self.anti0) @ self.params
        f0 = self.basis(0.0) @ self.params
        return np.where(t > 1e-9, integ / np.maximum(t, 1e-9), f0)


# ---------------------------------------------------------------- все методы на один день
METHODS = ["bootstrap", "bootstrap_1y", "ns", "svensson", "ns_fix", "ns_anchor", "spline"]


def fit_one(m, bonds, warm=None, only_warm=False, short_rate=None):
    """Одна кривая методом m. warm — параметры этого метода (например, с полной выборки) для старта.
    only_warm=False — мульти-старт добавляется к warm (честнее для leave-one-out)."""
    if m == "bootstrap":
        return Bootstrap(bonds)
    if m == "bootstrap_1y":
        return Bootstrap(bonds, bucket=1.0)
    if m == "ns":
        return Parametric(bonds, "ns", warm, only_warm)
    if m == "svensson":
        return Parametric(bonds, "sv", warm, only_warm)
    if m == "ns_fix":
        return Parametric(bonds, "nsfix", warm, only_warm)
    if m == "ns_anchor":
        if short_rate is None:
            return None
        return Parametric(bonds, "ns", warm, only_warm, short_rate=short_rate, short_band=ANCHOR_BAND)
    if m == "spline":
        return Spline(bonds, x0=warm)
    raise ValueError(m)


ANCHOR_BAND = 0.005  # ±0,5 п.п. вокруг RUONIA для мгновенной ставки в нуле


def fit_all(bonds, warm=None, short_rate=None, methods=METHODS):
    """Все собственные кривые на один день. short_rate — RUONIA (непрерывная, доли) для якоря."""
    warm = warm or {}
    out = {}
    for m in methods:
        f = fit_one(m, bonds, warm.get(m), short_rate=short_rate)
        if f is not None:
            out[m] = f
    return out
