"""Mock-payload tests for broker adapters (no credentials / network needed)."""
import sys, types, datetime as dt, json
import pandas as pd, numpy as np
import brokers as B, datahub as H

# ---- Dhan
class R:
    def __init__(s, j, code=200): s._j, s.status_code, s.text = j, code, json.dumps(j)[:100]
    def json(s): return s._j
base=pd.Timestamp("2026-10-01 03:45", tz="UTC").timestamp()   # 09:15 IST
calls=[]
def fake_post(url, json=None, headers=None, timeout=0):
    calls.append((url, json))
    if url.endswith("/charts/historical"):
        n=5; return R({"open":[1,2,3,4,5],"high":[2,3,4,5,6],"low":[0.5]*5,"close":[1.5,2.5,3.5,4.5,5.5],"volume":[10]*5,"timestamp":[base+86400*i for i in range(n)]})
    if url.endswith("/charts/intraday"):
        return R({"open":[1,2],"high":[2,3],"low":[1,1],"close":[2,3],"volume":[5,5],"timestamp":[base,base+900]})
    if url.endswith("/optionchain/expirylist"): return R({"data":["2026-10-06","2026-10-13"]})
    if url.endswith("/optionchain"): return R({"data":{"last_price":25000.5,"oc":{"25000.000000":{"ce":{"oi":1000,"previous_oi":900},"pe":{"oi":2000,"previous_oi":2100}},"25050.000000":{"ce":{"oi":500,"previous_oi":500},"pe":{"oi":100,"previous_oi":50}}}}})
import requests; requests.post=fake_post
d=B.Dhan({"client_id":"1","access_token":"x"})
df=d.history("NIFTY 50","index","1d",dt.datetime(2026,9,1),dt.datetime(2026,10,5))
assert len(df)==5 and df.index[0].hour==9 and df.index[0].minute==15, df.index[0]
assert calls[-1][1]["securityId"]=="13" and calls[-1][1]["exchangeSegment"]=="IDX_I" and calls[-1][1]["instrument"]=="INDEX"
di=d.history("NIFTY 50","index","15m",dt.datetime(2026,7,1),dt.datetime(2026,10,5)); assert len(di)>=2 and calls[-1][1]["interval"]=="15"
oc=d.option_chain("NIFTY 50"); st=__import__("engine").chain_stats(oc)
assert oc["expiry"]=="2026-10-06" and st["res"]==25000 and st["sup"]==25000 and oc["ce"][25000]==(1000,100) and oc["pe"][25000]==(2000,-100)
print("Dhan parse OK")

# ---- Angel One
class FakeSC:
    def __init__(s, api_key=None): pass
    def generateSession(s,a,b,c): return {"status":True}
    def getCandleData(s,p):
        assert p["interval"]=="FIFTEEN_MINUTE" and p["exchange"]=="NSE" and p["symboltoken"]=="99926000"
        return {"status":True,"data":[["2026-10-01T09:15:00+05:30",100,101,99,100.5,1000],["2026-10-01T09:30:00+05:30",100.5,102,100,101,900]]}
sys.modules["SmartApi"]=types.SimpleNamespace(SmartConnect=FakeSC)
sys.modules["pyotp"]=types.SimpleNamespace(TOTP=lambda s: types.SimpleNamespace(now=lambda:"123456"))
a=B.AngelOne({"api_key":"k","client_code":"c","pin":"1","totp_secret":"s"}); a.login()
df=a.history("NIFTY 50","index","15m",dt.datetime(2026,9,1),dt.datetime(2026,10,5)); assert len(df)==2 and df.index[0].hour==9 and df.index[0].minute==15
print("Angel parse OK")

# ---- Breeze
class FakeBZ:
    def __init__(s, api_key=None): pass
    def generate_session(s, api_secret=None, session_token=None): pass
    def get_names(s, exchange_code=None, stock_code=None): return {"isec_stock_code":"RELIND"}
    def get_historical_data_v2(s, **k):
        assert k["stock_code"] in ("RELIND","NIFTY") and k["product_type"]=="cash"
        return {"Success":[{"datetime":"2026-10-01 09:15:00","open":"10","high":"11","low":"9","close":"10.5","volume":"100"},{"datetime":"2026-10-01 09:20:00","open":"10.5","high":"11","low":"10","close":"10.8","volume":"50"}]}
sys.modules["breeze_connect"]=types.SimpleNamespace(BreezeConnect=FakeBZ)
b=B.Breeze({"api_key":"k","api_secret":"s","session_token":"t"}); b.login()
df=b.history("RELIANCE","stock","5m",dt.datetime(2026,9,25),dt.datetime(2026,10,5)); assert len(df)>=2 and df["Close"].iloc[0]==10.5
assert len(b.history("RELIANCE","stock","15m",dt.datetime(2026,9,25),dt.datetime(2026,10,5)))==0   # no native 15m -> hub resamples 5m
print("Breeze parse OK")

# ---- hub: broker fails -> falls to next source, resample works
d5=H.demo_intraday("TCS",5)
r=H.resample_intraday(d5,"75min"); assert len(r)==5*5, len(r)
cfg=H.Cfg("auto","Dhan",(("client_id",""),("access_token","")))
data,src,msgs=H.get_daily(["TCS"],"stock",H.Cfg("demo")); assert src["TCS"]=="DEMO"
print("hub OK; broker-login failure message:", H._broker(cfg)[1])
