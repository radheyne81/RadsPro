"""
brokers.py - Broker API adapters: Dhan, Angel One (SmartAPI), ICICI Direct (Breeze).

Every adapter exposes the same interface:
    login()                                  -> raises on failure
    history(sym, kind, interval, start, end) -> DataFrame[Open,High,Low,Close,Volume] (IST, tz-naive) or empty
    option_chain(index_sym)                  -> {"symbol","expiry","spot","ce":{k:(oi,chg)},"pe":{k:(oi,chg)}} or None
    resolve(sym, kind)                       -> broker-specific instrument id (or None)

interval: "1d", "1m", "5m", "15m", "30m", "60m"
kind    : "stock" | "index"

IMPORTANT - verification status
  These adapters follow each broker's published API shape, but they have NOT been run against a live
  account by the author (no credentials in the build sandbox). Parsing is unit-tested against mocked
  payloads (see test_brokers.py). Use  Data Sources -> "Test connection"  to confirm with your keys.
  Anything a broker cannot resolve is transparently served by the next source in the fallback chain,
  and the Provenance table shows which source produced each series.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import time

import numpy as np
import pandas as pd

OHLC = ["Open", "High", "Low", "Close", "Volume"]
CACHE = os.path.join(os.path.expanduser("~"), ".nse_terminal_cache")
os.makedirs(CACHE, exist_ok=True)


class BrokerError(RuntimeError):
    pass


def _frame(ts, o, h, l, c, v) -> pd.DataFrame:
    v = np.zeros(len(c)) if np.isscalar(v) else v
    cols = {k: np.asarray(x) for k, x in zip(OHLC, (o, h, l, c, v))}   # .values: avoid index-alignment NaNs
    df = pd.DataFrame(cols, index=pd.DatetimeIndex(pd.to_datetime(np.asarray(ts) if not hasattr(ts, "tz") else ts)))
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return df[~df.index.duplicated()].sort_index()


def _chunks(start: dt.datetime, end: dt.datetime, span_days: int):
    cur = start
    while cur < end:
        nxt = min(cur + dt.timedelta(days=span_days), end)
        yield cur, nxt
        cur = nxt + dt.timedelta(seconds=1)


def _cached_download(url: str, name: str, max_age_h: int = 20) -> bytes | None:
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_h * 3600:
        return open(path, "rb").read()
    try:
        import requests
        r = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and r.content:
            open(path, "wb").write(r.content)
            return r.content
    except Exception:
        pass
    return open(path, "rb").read() if os.path.exists(path) else None


# index display name -> candidate names in each broker's master
INDEX_ALIASES = {
    "NIFTY 50": ["NIFTY", "NIFTY 50", "NIFTY50"], "BANK NIFTY": ["BANKNIFTY", "NIFTY BANK", "NIFTY BANK INDEX"],
    "SENSEX": ["SENSEX", "BSE SENSEX"], "NIFTY IT": ["NIFTY IT", "NIFTYIT", "CNXIT"], "NIFTY AUTO": ["NIFTY AUTO", "CNXAUTO"],
    "NIFTY FMCG": ["NIFTY FMCG", "CNXFMCG"], "NIFTY METAL": ["NIFTY METAL", "CNXMETAL"], "NIFTY PHARMA": ["NIFTY PHARMA", "CNXPHARMA"],
    "NIFTY REALTY": ["NIFTY REALTY", "CNXREALTY"], "NIFTY ENERGY": ["NIFTY ENERGY", "CNXENERGY"],
    "NIFTY PSU BANK": ["NIFTY PSU BANK", "CNXPSUBANK", "NIFTYPSUBANK"], "NIFTY MEDIA": ["NIFTY MEDIA", "CNXMEDIA"],
    "NIFTY INFRA": ["NIFTY INFRA", "NIFTY INFRASTRUCTURE", "CNXINFRA"],
}


# =====================================================================================
# DHAN  (REST v2)  - needs client_id + access_token   (https://dhanhq.co/docs/v2/)
# =====================================================================================
class Dhan:
    name = "Dhan"
    BASE = "https://api.dhan.co/v2"
    # well-known index security ids (IDX_I)
    KNOWN_IDX = {"NIFTY 50": ("13", "IDX_I"), "BANK NIFTY": ("25", "IDX_I"), "SENSEX": ("51", "IDX_I")}
    INTERVAL = {"1m": "1", "5m": "5", "15m": "15", "30m": "25", "60m": "60"}

    def __init__(self, creds: dict):
        self.client_id = str(creds.get("client_id", "")).strip(); self.token = str(creds.get("access_token", "")).strip()
        self._master = None

    def _h(self):
        return {"access-token": self.token, "client-id": self.client_id, "Content-Type": "application/json", "Accept": "application/json"}

    def _post(self, path, body):
        import requests
        r = requests.post(self.BASE + path, json=body, headers=self._h(), timeout=30)
        if r.status_code != 200:
            raise BrokerError(f"Dhan {path} HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def login(self):
        if not self.client_id or not self.token:
            raise BrokerError("Dhan: client_id and access_token required")
        import requests
        r = requests.get(self.BASE + "/profile", headers=self._h(), timeout=20)
        if r.status_code != 200:
            raise BrokerError(f"Dhan login check failed HTTP {r.status_code}: {r.text[:200]}")

    def _load_master(self):
        if self._master is None:
            body = _cached_download("https://images.dhan.co/api-data/api-scrip-master.csv", "dhan_master.csv")
            if not body:
                raise BrokerError("Dhan: could not download scrip master")
            self._master = pd.read_csv(io.BytesIO(body), low_memory=False)
        return self._master

    @staticmethod
    def _col(df, *names):
        for n in names:
            if n in df.columns:
                return n
        return None

    def resolve(self, sym, kind):
        if kind == "index" and sym in self.KNOWN_IDX:
            return self.KNOWN_IDX[sym]
        m = self._load_master()
        ex, seg = self._col(m, "SEM_EXM_EXCH_ID"), self._col(m, "SEM_SEGMENT")
        inst, ts = self._col(m, "SEM_INSTRUMENT_NAME"), self._col(m, "SEM_TRADING_SYMBOL")
        cs, sid, ser = self._col(m, "SEM_CUSTOM_SYMBOL"), self._col(m, "SEM_SMST_SECURITY_ID"), self._col(m, "SEM_SERIES")
        if not (ex and inst and ts and sid):
            raise BrokerError("Dhan: unexpected scrip-master layout")
        if kind == "stock":
            q = m[(m[ex] == "NSE") & (m[inst] == "EQUITY") & (m[ts].astype(str).str.upper() == sym.upper())]
            if ser and len(q) > 1:
                q = q[q[ser].astype(str) == "EQ"] if (q[ser].astype(str) == "EQ").any() else q
            return (str(int(q.iloc[0][sid])), "NSE_EQ") if len(q) else None
        q = m[(m[inst] == "INDEX")]
        for cand in INDEX_ALIASES.get(sym, [sym]):
            hit = q[(q[ts].astype(str).str.upper() == cand.upper()) | ((q[cs].astype(str).str.upper() == cand.upper()) if cs else False)]
            if len(hit):
                return str(int(hit.iloc[0][sid])), "IDX_I"
        return None

    def history(self, sym, kind, interval, start, end):
        rid = self.resolve(sym, kind)
        if not rid:
            return pd.DataFrame()
        sid, seg = rid
        instrument = "INDEX" if kind == "index" else "EQUITY"
        frames = []
        if interval == "1d":
            j = self._post("/charts/historical", {"securityId": sid, "exchangeSegment": seg, "instrument": instrument, "expiryCode": 0,
                                                  "oi": False, "fromDate": start.strftime("%Y-%m-%d"), "toDate": end.strftime("%Y-%m-%d")})
            frames.append(self._parse(j))
        else:
            for a, b in _chunks(start, end, 85):          # Dhan intraday: max 90 days / call
                j = self._post("/charts/intraday", {"securityId": sid, "exchangeSegment": seg, "instrument": instrument,
                                                    "interval": self.INTERVAL[interval], "oi": False,
                                                    "fromDate": a.strftime("%Y-%m-%d %H:%M:%S"), "toDate": b.strftime("%Y-%m-%d %H:%M:%S")})
                frames.append(self._parse(j)); time.sleep(0.3)
        frames = [f for f in frames if len(f)]
        return pd.concat(frames).sort_index() if frames else pd.DataFrame()

    @staticmethod
    def _parse(j) -> pd.DataFrame:
        d = j.get("data", j)
        if not d or "close" not in d or not len(d["close"]):
            return pd.DataFrame()
        ts = pd.to_datetime(d.get("timestamp", d.get("start_Time")), unit="s", utc=True) if "timestamp" in d else pd.to_datetime(d["start_Time"])
        df = _frame(ts, d["open"], d["high"], d["low"], d["close"], d.get("volume", [0] * len(d["close"])))
        return df

    def option_chain(self, sym):
        rid = self.resolve(sym if sym in INDEX_ALIASES else "NIFTY 50", "index")
        if not rid:
            return None
        exps = self._post("/optionchain/expirylist", {"UnderlyingScrip": int(rid[0]), "UnderlyingSeg": rid[1]}).get("data", [])
        if not exps:
            return None
        exp = sorted(exps)[0]
        j = self._post("/optionchain", {"UnderlyingScrip": int(rid[0]), "UnderlyingSeg": rid[1], "Expiry": exp}).get("data", {})
        oc = j.get("oc", {}); ce, pe = {}, {}
        for k, v in oc.items():
            strike = int(float(k))
            if v.get("ce"):
                ce[strike] = (v["ce"].get("oi", 0) or 0, (v["ce"].get("oi", 0) or 0) - (v["ce"].get("previous_oi", 0) or 0))
            if v.get("pe"):
                pe[strike] = (v["pe"].get("oi", 0) or 0, (v["pe"].get("oi", 0) or 0) - (v["pe"].get("previous_oi", 0) or 0))
        return {"symbol": sym, "expiry": exp, "spot": j.get("last_price"), "ce": ce, "pe": pe}


# =====================================================================================
# ANGEL ONE SmartAPI  - needs api_key, client_code, pin, totp_secret   (pip install smartapi-python pyotp)
# =====================================================================================
class AngelOne:
    name = "Angel One"
    INTERVAL = {"1d": "ONE_DAY", "1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE", "30m": "THIRTY_MINUTE", "60m": "ONE_HOUR"}
    MAX_DAYS = {"1d": 1900, "1m": 25, "5m": 95, "15m": 190, "30m": 190, "60m": 380}
    KNOWN_IDX = {"NIFTY 50": "99926000", "BANK NIFTY": "99926009"}
    MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

    def __init__(self, creds: dict):
        self.c = {k: str(v).strip() for k, v in creds.items()}; self.api = None; self._master = None

    def login(self):
        try:
            from SmartApi import SmartConnect
            import pyotp
        except ImportError as e:
            raise BrokerError("Angel One needs:  pip install smartapi-python pyotp") from e
        for k in ("api_key", "client_code", "pin", "totp_secret"):
            if not self.c.get(k):
                raise BrokerError(f"Angel One: {k} required")
        self.api = SmartConnect(api_key=self.c["api_key"])
        r = self.api.generateSession(self.c["client_code"], self.c["pin"], pyotp.TOTP(self.c["totp_secret"]).now())
        if not r or not r.get("status"):
            raise BrokerError(f"Angel One login failed: {r.get('message') if r else 'no response'}")

    def _load_master(self):
        if self._master is None:
            body = _cached_download(self.MASTER_URL, "angel_master.json")
            if not body:
                raise BrokerError("Angel One: could not download instrument master")
            import json
            self._master = pd.DataFrame(json.loads(body))
        return self._master

    def resolve(self, sym, kind):
        if kind == "index" and sym in self.KNOWN_IDX:
            return self.KNOWN_IDX[sym], "NSE"
        m = self._load_master()
        if kind == "stock":
            q = m[(m["exch_seg"] == "NSE") & (m["symbol"].astype(str).str.upper() == f"{sym.upper()}-EQ")]
            return (str(q.iloc[0]["token"]), "NSE") if len(q) else None
        q = m[(m["exch_seg"].isin(["NSE", "BSE"])) & (m["instrumenttype"].astype(str).str.contains("IDX", na=False))]
        for cand in INDEX_ALIASES.get(sym, [sym]):
            hit = q[q["name"].astype(str).str.upper() == cand.upper()]
            if len(hit):
                return str(hit.iloc[0]["token"]), hit.iloc[0]["exch_seg"]
        return None

    def history(self, sym, kind, interval, start, end):
        rid = self.resolve(sym, kind)
        if not rid:
            return pd.DataFrame()
        tok, exch = rid; frames = []
        for a, b in _chunks(start, end, self.MAX_DAYS[interval]):
            r = self.api.getCandleData({"exchange": exch, "symboltoken": tok, "interval": self.INTERVAL[interval],
                                        "fromdate": a.strftime("%Y-%m-%d %H:%M"), "todate": b.strftime("%Y-%m-%d %H:%M")})
            frames.append(self._parse(r)); time.sleep(0.4)        # ~3 req/s limit
        frames = [f for f in frames if len(f)]
        return pd.concat(frames).sort_index() if frames else pd.DataFrame()

    @staticmethod
    def _parse(r) -> pd.DataFrame:
        rows = (r or {}).get("data") or []
        if not rows:
            return pd.DataFrame()
        a = list(zip(*rows))                                      # [ts, o, h, l, c, v]
        ts = pd.to_datetime([str(x)[:19] for x in a[0]])          # ISO with +05:30 -> naive IST
        return _frame(ts, a[1], a[2], a[3], a[4], a[5])

    def option_chain(self, sym):
        return None   # SmartAPI has no consolidated chain endpoint; hub falls back to NSE live / demo


# =====================================================================================
# ICICI DIRECT Breeze  - needs api_key, api_secret, session_token (daily)   (pip install breeze-connect)
# =====================================================================================
class Breeze:
    name = "ICICI Breeze"
    INTERVAL = {"1d": "1day", "1m": "1minute", "5m": "5minute", "30m": "30minute"}   # Breeze has no 15m / 60m
    KNOWN_IDX = {"NIFTY 50": "NIFTY", "BANK NIFTY": "CNXBAN"}

    def __init__(self, creds: dict):
        self.c = {k: str(v).strip() for k, v in creds.items()}; self.api = None; self._codes = {}

    def login(self):
        try:
            from breeze_connect import BreezeConnect
        except ImportError as e:
            raise BrokerError("ICICI Breeze needs:  pip install breeze-connect") from e
        for k in ("api_key", "api_secret", "session_token"):
            if not self.c.get(k):
                raise BrokerError(f"ICICI Breeze: {k} required (session_token is regenerated every day from the login URL)")
        self.api = BreezeConnect(api_key=self.c["api_key"])
        self.api.generate_session(api_secret=self.c["api_secret"], session_token=self.c["session_token"])

    def resolve(self, sym, kind):
        if kind == "index":
            return self.KNOWN_IDX.get(sym)
        if sym in self._codes:
            return self._codes[sym]
        try:
            r = self.api.get_names(exchange_code="NSE", stock_code=sym)
            code = (r or {}).get("isec_stock_code") or None
        except Exception:
            code = None
        self._codes[sym] = code
        return code

    def history(self, sym, kind, interval, start, end):
        if interval not in self.INTERVAL:
            return pd.DataFrame()                                # 15m/60m not offered -> next source (hub can resample 5m)
        code = self.resolve(sym, kind)
        if not code:
            return pd.DataFrame()
        frames = []
        span = {"1d": 900, "5m": 12, "1m": 2, "30m": 60}[interval]    # Breeze returns <=1000 candles per call
        for a, b in _chunks(start, end, span):
            r = self.api.get_historical_data_v2(interval=self.INTERVAL[interval], from_date=a.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                                                to_date=b.strftime("%Y-%m-%dT%H:%M:%S.000Z"), stock_code=code, exchange_code="NSE",
                                                product_type="cash")
            frames.append(self._parse(r)); time.sleep(0.3)
        frames = [f for f in frames if len(f)]
        return pd.concat(frames).sort_index() if frames else pd.DataFrame()

    @staticmethod
    def _parse(r) -> pd.DataFrame:
        rows = (r or {}).get("Success") or []
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "datetime" not in df.columns:
            return pd.DataFrame()
        return _frame(pd.to_datetime(df["datetime"]), df["open"], df["high"], df["low"], df["close"], df.get("volume", 0))

    def option_chain(self, sym):
        return None


PROVIDERS = {"Dhan": Dhan, "Angel One": AngelOne, "ICICI Breeze": Breeze}
CRED_FIELDS = {
    "Dhan": [("client_id", "Client ID", False), ("access_token", "Access token (from web.dhan.co → API)", True)],
    "Angel One": [("api_key", "SmartAPI key", True), ("client_code", "Client code", False), ("pin", "PIN", True), ("totp_secret", "TOTP secret (QR text)", True)],
    "ICICI Breeze": [("api_key", "API key", True), ("api_secret", "API secret", True), ("session_token", "Session token (daily)", True)],
}
_SESSIONS: dict = {}


def get_provider(name: str, creds: dict):
    """Logged-in provider, cached per credential set for the life of the process."""
    key = (name, tuple(sorted(creds.items())))
    if key not in _SESSIONS:
        p = PROVIDERS[name](creds); p.login(); _SESSIONS[key] = p
    return _SESSIONS[key]
