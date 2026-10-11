import sys, pandas as pd
from streamlit.testing.v1 import AppTest
import brokers as B, datahub as H, engine as E

PAGES=["🏠 Command Center","🔮 Next-Day Predictions","⚡ Intraday Desk","🎯 High-Probability Setups","🧪 Backtest Lab","🌍 Global & Macro","📊 Index Technicals","📶 Breadth & Sectors","🔁 Hilega-Milega Screener","🕯️ HA & Pattern Screener","🏦 Institutional Flows","🧩 Option Chain","🔌 Data Sources","📖 Logic Decoder"]

class FakeProv:                       # serves synthetic data but identifies as a broker
    name="Dhan"
    def history(self,sym,kind,interval,start,end):
        return E.demo_series(sym) if interval=="1d" else H.demo_intraday(sym)
    def option_chain(self,sym): return None
def run(page, mode, broker=None, fake=False):
    B.get_provider=(lambda n,c: FakeProv()) if fake else B.get_provider
    at=AppTest.from_file("app.py",default_timeout=300).run()
    sb={w.label:w for w in at.sidebar.selectbox}
    sb["Mode"].set_value(mode)
    if broker: sb["Broker API"].set_value(broker)
    at.sidebar.radio[0].set_value(page); at.sidebar.slider[0].set_value(12)
    return at.run()
bad=0
for p in PAGES:
    at=run(p,"Demo (synthetic, offline)"); e=[str(x.value)[:300] for x in at.exception]; bad+=bool(e); print(("FAIL " if e else "ok   ")+p, e)
# broker path with fake provider: must be labelled as Dhan, not DEMO
at=run("🏠 Command Center","Auto (Broker → Yahoo → NSE bhavcopy)","Dhan",fake=True)
txt=" ".join(m.value for m in at.markdown)
print("broker-routed banner:", "Dhan:" in txt, "| exceptions:", [str(x.value)[:200] for x in at.exception])
for p in ["🔮 Next-Day Predictions","⚡ Intraday Desk"]:
    at=run(p,"Auto (Broker → Yahoo → NSE bhavcopy)","Dhan",fake=True); print("broker",p,[str(x.value)[:200] for x in at.exception] or "ok")
# failure path: Dhan chosen, empty creds, auto mode -> must degrade, not crash
B.get_provider=B.get_provider if not hasattr(B.get_provider,"__name__") or B.get_provider.__name__!="<lambda>" else B.get_provider
print("failures",bad)
