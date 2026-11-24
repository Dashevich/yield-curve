"""
Восстановление кривой бескупонных ставок по ОФЗ-ПД.

Модуль содержит:
  * загрузку данных из data/raw (итоги торгов, графики купонов, КБД Мосбиржи);
  * КБД Мосбиржи (G-кривая: Нельсон–Сигель + 9 гауссовых поправок);
  * бутстрэп (кусочно-постоянные форварды между погашениями);
  * Нельсон–Сигель и Свенссон с калибровкой по ошибкам в доходности.

Ставки внутри модуля — непрерывно начисляемые, в долях (0.15 = 15%).
Время — в годах ACT/365 от даты расчётов (T+1).
"""

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")


# ---------------------------------------------------------------- данные
def load_history():
    """Итоги торгов ОФЗ-ПД (SU25*/SU26*, постоянный купон) за все дни."""
    frames = []
    for f in sorted(glob.glob(os.path.join(ROOT, "history", "*.csv"))):
        if os.path.getsize(f) < 10:  # выходной день — пустой файл
            continue
        frames.append(pd.read_csv(f, encoding="utf-8"))
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


def bond_price(bond, zero_fn):
    return float(np.sum(bond.cf * np.exp(-zero_fn(bond.t) * bond.t)))


def ytm_cont(t, cf, price):
    f = lambda y: np.sum(cf * np.exp(-y * t)) - price
    return brentq(f, -0.5, 2.0)


def _clean_price(r):
    """Средневзвешенная цена, если она внутри дневного диапазона [LOW, HIGH], иначе цена закрытия.

    WAPRICE иногда выходит за диапазон стакана — в неё попадают внесистемные сделки
    по нерыночной цене (напр. 24.07.2026 SU26212: WAPRICE 97,49 при HIGH 91,39).
    Такие цены дают выбросы доходности в сотни б.п. и портят кривую на коротком конце.
    """
    wap, lo, hi = r.WAPRICE, r.LOW, r.HIGH
    if pd.isna(wap):
        return r.CLOSE
    if pd.notna(lo) and pd.notna(hi) and not (lo <= wap <= hi) and pd.notna(r.CLOSE):
        return r.CLOSE
    return wap


def market_on(h_day, cashflows, settle, min_trades=5, min_years=0.25):
    """Список облигаций дня с ценами и потоками после даты расчётов.

    Фильтры: есть цена (средневзвешенная в пределах дня, иначе закрытия), >= min_trades сделок,
    срок до погашения >= min_years (короткие выпуски шумные из-за НКД и ликвидности).
    """
    bonds = []
    for r in h_day.itertuples():
        price = _clean_price(r)
        if pd.isna(price) or pd.isna(r.ACCINT) or (r.NUMTRADES or 0) < min_trades:
            continue
        cf = cashflows[r.SECID]
        cf = cf[cf.date > settle]
        if cf.empty:
            continue
        t = (cf.date - settle).dt.days.values / 365.0
        if t[-1] < min_years:
            continue
        face = r.FACEVALUE if pd.notna(r.FACEVALUE) else 1000.0
        dirty = price / 100 * face + r.ACCINT
        a = cf.amount.values.astype(float)
        try:
            y = ytm_cont(t, a, dirty)
        except ValueError:
            continue
        w = a * np.exp(-y * t)
        bonds.append(Bond(r.SECID, t, a, dirty, y, float(np.sum(w * t) / np.sum(w)),
                          float(t[-1]), float(r.VALUE or 0), int(r.NUMTRADES or 0)))
    bonds.sort(key=lambda b: b.maturity)
    return bonds


def yield_errors(bonds, zero_fn):
    """Ошибки модели в доходности, б.п. (модельная YTM минус рыночная)."""
    out = []
    for b in bonds:
        p = bond_price(b, zero_fn)
        out.append((ytm_cont(b.t, b.cf, p) - b.ytm) * 1e4)
    return np.array(out)


# ---------------------------------------------------------------- бутстрэп
class Bootstrap:
    """Кусочно-постоянные мгновенные форварды между погашениями облигаций.

    Узлы — сроки погашения; z(t)*t линейна между узлами (форвард постоянен).
    До первого узла форвард равен первому, за последним — постоянен (плоская экстраполяция).
    Каждая следующая облигация оценивается точно — подбирается форвард на её участке.
    Выпуски, погашающиеся ближе 30 дней к предыдущему узлу, пропускаются
    (иначе форвард на коротком участке «взрывается» от шума цен).
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
        for b in bonds:
            if b.maturity - self.knots[-1] < min_gap:
                continue
            self.knots.append(b.maturity)
            self.fwd.append(0.1)
            f = lambda x: self._set_last(x) or (bond_price(b, self.zero) - b.dirty)
            try:
                self.fwd[-1] = brentq(f, -0.5, 1.5)
            except ValueError:
                self.knots.pop(); self.fwd.pop()
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


class Parametric:
    """Калибровка NS/Свенссона: минимизируем ошибки цен, делённые на (цена × дюрация),
    т.е. приближённо ошибки в доходности. Мульти-старт по τ, т.к. задача невыпуклая.

    Ограничения (без них Свенссон вырождается: τ1 -> 0 даёт всплеск на коротком конце,
    τ1 ≈ τ2 — коллинеарность, β2 и β3 уходят на границы с разными знаками):
      * β0 (долгосрочный уровень) в [2%, 40%]; |β1| ≤ 50 п.п.; |β2|, |β3| ≤ 30 п.п.;
      * τ1 в [0.25, 10] лет; у Свенссона τ2 = τ1 + Δ, Δ в [1, 30] лет (горбы разнесены);
      * мгновенная ставка в нуле β0+β1 — не дальше SHORT_BAND от доходности самого
        короткого выпуска: внутри полосы штрафа нет, за её пределами — штраф,
        как за ошибку SHORT_W б.п. на каждый б.п. выхода.
    """

    SHORT_BAND = 0.02
    SHORT_W = 1.0
    HUBER_BP = 15.0  # порог функции Хьюбера для робастного варианта, б.п.

    def __init__(self, bonds, kind="ns", x0=None, only_x0=False, robust=False):
        """robust=True — функция потерь Хьюбера: ошибки больше HUBER_BP штрафуются линейно,
        а не квадратично, поэтому одна аномальная цена не тянет за собой всю кривую."""
        self.kind = kind
        self.fn = ns_zero if kind == "ns" else sv_zero
        self.fwd_fn = ns_forward if kind == "ns" else sv_forward
        T = [b.t for b in bonds]
        C = [b.cf for b in bonds]
        P = np.array([b.dirty for b in bonds])
        D = np.array([b.dur for b in bonds])
        y_long = bonds[-1].ytm
        y_short = bonds[0].ytm
        sv = kind != "ns"

        def to_params(q):  # внутренние переменные -> (β..., τ1, τ2)
            return np.r_[q[:5], q[4] + q[5]] if sv else q

        def from_params(p):
            p = np.asarray(p, float)
            return np.r_[p[:5], p[5] - p[4]] if sv else p

        def resid(q):
            p = to_params(q)
            model = np.array([np.sum(c * np.exp(-self.fn(p, t) * t)) for t, c in zip(T, C)])
            r = (model - P) / (P * D) * 1e4
            out = max(abs(p[0] + p[1] - y_short) - self.SHORT_BAND, 0.0)
            return np.append(r, self.SHORT_W * out * 1e4)

        if not sv:
            lo, hi = [0.02, -0.5, -0.3, 0.25], [0.4, 0.5, 0.3, 10]
            starts = [[y_long, y_short - y_long, 0.0, tau] for tau in (0.5, 1.5, 3, 6)]
        else:
            lo, hi = [0.02, -0.5, -0.3, -0.3, 0.25, 1.0], [0.4, 0.5, 0.3, 0.3, 10, 30]
            starts = [[y_long, y_short - y_long, 0.0, 0.0, t1, dt]
                      for t1, dt in ((0.5, 2.5), (1, 4), (1.5, 6), (3, 7), (0.8, 14))]
        if x0 is not None:
            w = list(from_params(x0))
            starts = [w] if only_x0 else [w] + starts
        best = None
        for s in starts:
            s = np.clip(s, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
            try:
                r = least_squares(resid, s, bounds=(lo, hi), method="trf", max_nfev=400,
                                  loss="huber" if robust else "linear", f_scale=self.HUBER_BP)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        self.params = to_params(best.x)
        self.cost = best.cost

    def zero(self, t):
        return self.fn(self.params, t)

    def forward(self, t):
        return self.fwd_fn(self.params, t)


# ---------------------------------------------------------------- сглаживающий сплайн (VRP)
class Spline:
    """Сглаживающий сплайн форвардной кривой с переменным штрафом (VRP: Waggoner 1997,
    Anderson & Sleath 2001 — метод Банка Англии).

    Мгновенный форвард f(t) — кубический B-сплайн с фиксированными узлами KNOTS;
    z(t)·t = ∫₀ᵗ f — линейна по коэффициентам. Минимизируем
        Σ (ошибка цены / (цена × дюрация))²  +  ∫ λ(t) f''(t)² dt,
        log λ(t) = L − (L − S)·exp(−t / MU).
    Штраф на кривизну мал на коротком конце (форма кривой может быстро меняться —
    как в стресс) и велик на длинном (там мало выпусков, «пилы» бутстрэпа нет).
    L, S подобраны по leave-one-out на 2024 H1 (до периода стресса), см. tune_spline.py.
    Лучшим оказался постоянный штраф L = S = −5 (т.е. вариант Fisher–Nychka–Zervos);
    варианты с меньшим штрафом на коротком конце хуже на 0,5–1 б.п.
    За самым длинным выпуском форвард постоянен (без экстраполяции наклона).
    """

    KNOTS = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30], float)
    L, S, MU = -5.0, -5.0, 1.0
    _cache = {}

    @classmethod
    def _basis(cls, tmax):
        """B-сплайны на узлах до самого длинного выпуска; дальше форвард постоянен."""
        from scipy.interpolate import BSpline
        kn = np.r_[cls.KNOTS[cls.KNOTS < tmax - 0.5], tmax]
        key = tuple(np.round(kn, 6))
        if key not in cls._cache:
            tk = np.r_[[kn[0]] * 3, kn, [kn[-1]] * 3]
            K = len(tk) - 4
            B = BSpline(tk, np.eye(K), 3, extrapolate=False)
            if len(cls._cache) > 2000:
                cls._cache.clear()
            cls._cache[key] = (B, B.antiderivative(), B.derivative(2))
        return cls._cache[key]

    def _A(self, t):
        """∫₀ᵗ B_k с плоским форвардом после tmax."""
        tc = np.minimum(t, self.tmax)
        out = self.A(tc) - self.A0
        return out + (t - tc)[:, None] * self.Bend[None, :]

    def __init__(self, bonds, x0=None, L=None, S=None, mu=None):
        L = self.L if L is None else L
        S = self.S if S is None else S
        mu = self.MU if mu is None else mu
        self.tmax = bonds[-1].maturity
        B, A, B2 = self._basis(self.tmax)
        self.B, self.A = B, A
        self.A0 = A(0.0)
        self.Bend = B(self.tmax - 1e-9)
        M = [self._A(b.t) for b in bonds]                      # ∫₀ᵗ B_k по каждому потоку
        CF = [b.cf for b in bonds]
        P = np.array([b.dirty for b in bonds])
        W = 1e4 / (P * np.array([b.dur for b in bonds]))
        g = np.linspace(0, self.tmax - 1e-9, 121)
        lam = np.exp(L - (L - S) * np.exp(-g / mu))
        Pen = B2(g) * np.sqrt(lam * (g[1] - g[0]))[:, None] * 1e4  # штраф — тоже в «б.п.»

        def resid(c):
            model = np.array([np.sum(cf * np.exp(-m @ c)) for m, cf in zip(M, CF)])
            return np.r_[(model - P) * W, Pen @ c]

        def jac(c):
            J = np.array([-(cf * np.exp(-m @ c)) @ m for m, cf in zip(M, CF)]) * W[:, None]
            return np.vstack([J, Pen])

        K = B.c.shape[0]
        c0 = np.full(K, np.mean([b.ytm for b in bonds]))
        if x0 is not None and len(x0) == K:  # тёплый старт, если набор узлов тот же
            c0 = np.asarray(x0, float)
        r = least_squares(resid, c0, jac=jac, method="lm", max_nfev=200)
        self.params = r.x
        self.cost = r.cost

    def zero(self, t):
        t = np.maximum(np.atleast_1d(np.asarray(t, float)), 1e-6)
        return (self._A(t) @ self.params) / t

    def forward(self, t):
        t = np.minimum(np.atleast_1d(np.asarray(t, float)), self.tmax - 1e-9)
        return self.B(np.maximum(t, 0.0)) @ self.params


def fit_all(bonds, warm=None):
    """Все собственные кривые на один день. warm — параметры прошлого дня для старта."""
    warm = warm or {}
    ns = Parametric(bonds, "ns", warm.get("ns"))
    return {
        "bootstrap": Bootstrap(bonds),
        "bootstrap_1y": Bootstrap(bonds, bucket=1.0),
        "ns": ns,
        "ns_robust": Parametric(bonds, "ns", ns.params, robust=True),
        "svensson": Parametric(bonds, "sv", warm.get("svensson")),
        "spline": Spline(bonds, warm.get("spline")),
    }
