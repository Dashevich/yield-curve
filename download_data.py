"""
Выгрузка данных по ОФЗ с Московской биржи (ISS API).

Что скачивает (в папку data/raw рядом со скриптом):
  history/<дата>.csv      — итоги торгов всеми ОФЗ (режим TQOB) за каждый день:
                            цены, НКД, доходности, обороты, даты погашения
  securities/<SECID>.csv  — описание каждого выпуска (частота купона, даты и т.д.)
  bondization/<SECID>_coupons.csv, _amortizations.csv, _offers.csv
                          — график купонов, амортизаций и оферт по выпуску
  zcyc/<дата>.json        — КБД Мосбиржи (параметры кривой) на каждый торговый день
  rgbi.csv                — индекс гособлигаций RGBI (Мосбиржа)
  keyrate.html, ruonia.html, calendar.html — ключевая ставка, RUONIA и даты заседаний
                          (сайт Банка России)

Нужен только Python 3.8+ — никаких дополнительных библиотек.
Запуск:   python download_data.py
          python download_data.py 2024-01-01 2026-10-31   (свой период)

Скрипт можно прерывать (Ctrl+C) и запускать снова: уже скачанное он пропускает.
Ошибки пишутся в data/raw/errors.log, скачивание при этом продолжается.
"""

import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

# ISS_BASE=http://iss.moex.com/iss/ — если https до биржи не проходит (у ISS публичные данные, http тоже работает)
BASE = os.environ.get("ISS_BASE", "https://iss.moex.com/iss/")
START = "2024-01-01"
END = "2026-10-31"
PAUSE = 0.25  # пауза между запросами, чтобы не нагружать биржу

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
DIRS = {k: os.path.join(ROOT, k) for k in ("history", "securities", "bondization", "zcyc")}
ERRLOG = os.path.join(ROOT, "errors.log")


def log_error(msg):
    print("  ОШИБКА:", msg)
    with open(ERRLOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")


def get_json(path, params=None, retries=4):
    params = dict(params or {})
    params.setdefault("iss.meta", "off")
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ofz-curve-study/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            time.sleep(PAUSE)
            return data
        except Exception as e:  # сеть, таймаут, 5xx
            if attempt == retries - 1:
                raise RuntimeError(f"{url}: {e}")
            time.sleep(2 * (attempt + 1))


def fetch_block(path, params, block):
    """Скачивает таблицу `block` целиком, перелистывая страницы (ISS отдаёт порциями)."""
    cols, rows, start, prev_page = None, [], 0, None
    while True:
        js = get_json(path, dict(params, start=start))
        b = js.get(block)
        if not b or not b.get("data"):
            break
        cols, page = b["columns"], b["data"]
        if page == prev_page:  # сервер не листает — дальше то же самое
            break
        rows.extend(page)
        prev_page = page
        cur = js.get(block + ".cursor")
        if cur and cur.get("data"):
            c = dict(zip(cur["columns"], cur["data"][0]))
            start = c["INDEX"] + c["PAGESIZE"]
            if start >= c["TOTAL"]:
                break
        else:
            start += len(page)
    return cols, rows


def write_csv(path, cols, rows):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if cols:
            w.writerow(cols)
            w.writerows(rows)
    os.replace(tmp, path)  # файл появляется только целиком


def days(start, end):
    d, e = date.fromisoformat(start), date.fromisoformat(end)
    while d <= e:
        yield d.isoformat()
        d += timedelta(days=1)


# ---------- 1. Итоги торгов по дням ----------
def download_history(start, end):
    print(f"\n[1/6] Итоги торгов ОФЗ {start} — {end}")
    trading_days, secids = [], set()
    all_days = list(days(start, end))
    for i, d in enumerate(all_days, 1):
        path = os.path.join(DIRS["history"], d + ".csv")
        if not os.path.exists(path):
            try:
                cols, rows = fetch_block(
                    "history/engines/stock/markets/bonds/boards/TQOB/securities.json",
                    {"date": d}, "history")
                write_csv(path, cols, rows)  # пустой файл = выходной, больше не спрашиваем
                print(f"  {d}: {len(rows)} выпусков   ({i}/{len(all_days)})")
            except Exception as e:
                log_error(f"history {d}: {e}")
                continue
        with open(path, encoding="utf-8") as f:
            r = list(csv.DictReader(f))
        if r:
            trading_days.append(d)
            secids.update(x["SECID"] for x in r)
    print(f"  Торговых дней: {len(trading_days)}, разных выпусков: {len(secids)}")
    return trading_days, sorted(secids)


# ---------- 2. Описание выпусков ----------
def download_securities(secids):
    print(f"\n[2/6] Описание выпусков ({len(secids)})")
    for i, s in enumerate(secids, 1):
        path = os.path.join(DIRS["securities"], s + ".csv")
        if os.path.exists(path):
            continue
        try:
            js = get_json(f"securities/{s}.json", {"iss.only": "description"})
            b = js.get("description", {})
            write_csv(path, b.get("columns"), b.get("data", []))
            print(f"  {s}  ({i}/{len(secids)})")
        except Exception as e:
            log_error(f"securities {s}: {e}")


# ---------- 3. Купоны, амортизации, оферты ----------
def download_bondization(secids):
    print(f"\n[3/6] Графики купонов ({len(secids)})")
    for i, s in enumerate(secids, 1):
        for block in ("coupons", "amortizations", "offers"):
            path = os.path.join(DIRS["bondization"], f"{s}_{block}.csv")
            if os.path.exists(path):
                continue
            try:
                cols, rows = fetch_block(
                    f"statistics/engines/stock/markets/bonds/bondization/{s}.json",
                    {"iss.only": f"{block},{block}.cursor"}, block)
                write_csv(path, cols, rows)
            except Exception as e:
                log_error(f"bondization {s} {block}: {e}")
        print(f"  {s}  ({i}/{len(secids)})")


# ---------- 4. КБД Мосбиржи ----------
def download_zcyc(trading_days):
    print(f"\n[4/6] КБД Мосбиржи ({len(trading_days)} дней)")
    for i, d in enumerate(trading_days, 1):
        path = os.path.join(DIRS["zcyc"], d + ".json")
        if os.path.exists(path):
            continue
        try:
            js = get_json("engines/stock/zcyc.json", {"date": d})
            tmp = path + ".part"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(js, f, ensure_ascii=False)
            os.replace(tmp, path)
            if i % 20 == 0 or i == len(trading_days):
                print(f"  {d}  ({i}/{len(trading_days)})")
        except Exception as e:
            log_error(f"zcyc {d}: {e}")


# ---------- 5. Индекс гособлигаций RGBI (для рыночного определения стресса) ----------
def download_rgbi(start, end):
    print("\n[5/6] Индекс RGBI")
    path = os.path.join(ROOT, "rgbi.csv")
    try:
        cols, rows = fetch_block("history/engines/stock/markets/index/securities/RGBI.json",
                                 {"from": start, "till": end}, "history")
        write_csv(path, cols, rows)
        print(f"  {len(rows)} дней")
    except Exception as e:
        log_error(f"rgbi: {e}")


# ---------- 6. Ключевая ставка и RUONIA с сайта Банка России ----------
CBR_PAGES = {
    "keyrate.html": "https://www.cbr.ru/hd_base/KeyRate/",
    "ruonia.html": "https://www.cbr.ru/hd_base/ruonia/dynamics/",
    "calendar.html": "https://www.cbr.ru/dkp/cal_mp/",  # даты заседаний по ключевой ставке
}


def download_cbr(start):
    """Сохраняем HTML-таблицы ЦБ как есть (разбор — в curves.load_cbr). Период — до сегодня:
    ключевая ставка нужна и после конца выборки, чтобы проверять годовые прогнозы."""
    print("\n[6/6] Ключевая ставка и RUONIA (cbr.ru)")
    d0 = date.fromisoformat(start) - timedelta(days=60)
    params = {"UniDbQuery.Posted": "True", "UniDbQuery.From": d0.strftime("%d.%m.%Y"),
              "UniDbQuery.To": date.today().strftime("%d.%m.%Y")}
    for name, url in CBR_PAGES.items():
        try:
            req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                         headers={"User-Agent": "Mozilla/5.0 ofz-curve-study/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                html = r.read().decode("utf-8")
            with open(os.path.join(ROOT, name), "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  {name}: {len(html) // 1024} КБ")
        except Exception as e:
            log_error(f"cbr {name}: {e}")


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else START
    end = sys.argv[2] if len(sys.argv) > 2 else END
    for p in DIRS.values():
        os.makedirs(p, exist_ok=True)
    print("Папка с данными:", ROOT)
    t0 = time.time()
    trading_days, secids = download_history(start, end)
    download_securities(secids)
    download_bondization(secids)
    download_zcyc(trading_days)
    download_rgbi(start, end)
    download_cbr(start)
    print(f"\nГотово за {(time.time() - t0) / 60:.1f} мин.")
    if os.path.exists(ERRLOG):
        print("Были ошибки — см. data/raw/errors.log. Просто запустите скрипт ещё раз,"
              " он докачает недостающее.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено. Запустите снова — продолжит с того же места.")
