"""NSE AI SMART TERMINAL v2 - Streamlit dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import brokers as B
import datahub as H
import engine as E
import intraday as I
import predict as P

st.set_page_config(page_title="NSE AI Smart Terminal", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

GREEN, RED, AMBER, CYAN, VIOLET, MUTED = "#22c55e", "#ef4444", "#f59e0b", "#22d3ee", "#a78bfa", "#94a3b8"
st.markdown(f"""
<style>
.block-container {{padding-top:1.2rem; max-width:1600px;}}
.hero {{background:linear-gradient(120deg,#0f2747 0%,#14305c 45%,#1b1f4b 100%);border:1px solid #1e3a5f;border-radius:16px;padding:18px 24px;margin-bottom:14px}}
.hero h1 {{margin:0;font-size:1.7rem;letter-spacing:-.02em}} .hero p {{margin:.2rem 0 0;color:{MUTED};font-size:.9rem}}
.card {{background:#111a2e;border:1px solid #1e2b45;border-radius:14px;padding:14px 16px;height:100%}}
.card .k {{color:{MUTED};font-size:.72rem;text-transform:uppercase;letter-spacing:.08em}}
.card .v {{font-size:1.55rem;font-weight:700;line-height:1.25}} .card .s {{color:{MUTED};font-size:.78rem}}
.logic {{background:#0e1a2f;border-left:4px solid {CYAN};border-radius:8px;padding:10px 14px;margin:6px 0;font-size:.88rem}}
.demo {{background:#4a3205;border:1px solid #f59e0b;color:#fde68a;border-radius:10px;padding:8px 14px;margin-bottom:10px;font-size:.88rem}}
.ok {{background:#06402b;border:1px solid #22c55e;color:#bbf7d0;border-radius:10px;padding:8px 14px;margin-bottom:10px;font-size:.88rem}}
</style>""", unsafe_allow_html=True)
IST = ZoneInfo("Asia/Kolkata")


# ================================================================ UI helpers
def card(k, v, s="", color="#e2e8f0", border=None):
    b = f"border-left:4px solid {border};" if border else ""
    st.markdown(f'<div class="card" style="{b}"><div class="k">{k}</div><div class="v" style="color:{color}">{v}</div><div class="s">{s}</div></div>', unsafe_allow_html=True)

def tone(x, pos=GREEN, neg=RED):
    return pos if x > 0 else neg if x < 0 else MUTED

def logic(txt):
    st.markdown(f'<div class="logic">{txt}</div>', unsafe_allow_html=True)

def _sty(df):
    return df if hasattr(df, "data") and hasattr(df, "map") and not isinstance(df, pd.DataFrame) else df.style

def style_pm(df, cols):
    def f(v):
        try:
            return f"color:{GREEN};font-weight:600" if v > 0 else f"color:{RED};font-weight:600" if v < 0 else ""
        except Exception:
            return ""
    s = _sty(df); return s.map(f, subset=[c for c in cols if c in s.data.columns])

def style_tags(df, cols):
    def f(v):
        s = str(v)
        if s in ("BULL", "LONG", "Bullish", "ROBUST EDGE", "PROBABLE EDGE", "A+", "A", "BULLISH", "STRONG BULL", "LEADING", "IMPROVING"): return "background:#06402b;color:#4ade80;font-weight:700"
        if s in ("BEAR", "SHORT", "Bearish", "NO EDGE", "DECAYING / OVERFIT", "D", "BEARISH", "STRONG BEAR", "LAGGING"): return "background:#4a1111;color:#f87171;font-weight:700"
        if s in ("NEUTRAL", "Neutral", "MARGINAL", "B", "C", "INSUFFICIENT DATA", "WEAKENING"): return "background:#4a3205;color:#fbbf24;font-weight:700"
        return ""
    s = _sty(df); return s.map(f, subset=[c for c in cols if c in s.data.columns])

def dark(fig, h=420):
    fig.update_layout(template="plotly_dark", height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.08), font=dict(size=12))
    fig.update_xaxes(gridcolor="#1e2b45"); fig.update_yaxes(gridcolor="#1e2b45")
    return fig

def banner(srcs: dict | None = None, msgs: list | None = None):
    srcs = srcs or {}
    demo = [k for k, v in srcs.items() if str(v).startswith("DEMO")]
    live = Counter(v for k, v in srcs.items() if not str(v).startswith("DEMO"))
    live_txt = " · ".join(f"{k}: {v}" for k, v in live.items()) or "none"
    if demo:
        names = ", ".join(demo[:6]) + ("…" if len(demo) > 6 else "")
        st.markdown(f'<div class="demo">⚠️ <b>PARTIAL / DEMO DATA</b> — {len(demo)} feed(s) are synthetic ({names}). Live feeds: {live_txt}. '
                    'Synthetic numbers are random: <b>do not trade on them.</b> Add a broker in the sidebar or run on your own PC.</div>', unsafe_allow_html=True)
    elif live:
        st.markdown('<div class="ok">✅ Live data — ' + live_txt + "</div>", unsafe_allow_html=True)
    for m in (msgs or [])[:4]:
        st.caption("• " + m)


# ================================================================ sidebar: data sources
def _secret(broker_key: str, field: str) -> str:
    env = os.getenv(f"{broker_key}_{field}".upper().replace(" ", "_"), "")
    try:
        return str(st.secrets.get("brokers", {}).get(broker_key, {}).get(field, env))
    except Exception:
        return env

MODES = {"Auto (Broker → Yahoo → NSE bhavcopy)": "auto", "Broker API only": "broker", "Yahoo Finance only (free)": "yahoo",
         "NSE bhavcopy only (free, official, daily)": "nse", "Demo (synthetic, offline)": "demo"}
PAGES = ["🏠 Command Center", "🔮 Next-Day Predictions", "⚡ Intraday Desk", "🎯 High-Probability Setups", "🧪 Backtest Lab", "🌍 Global & Macro",
         "📊 Index Technicals", "📶 Breadth & Sectors", "🔁 Hilega-Milega Screener", "🕯️ HA & Pattern Screener", "🏦 Institutional Flows",
         "🧩 Option Chain", "🔌 Data Sources", "📖 Logic Decoder"]
with st.sidebar:
    st.markdown("### ⚙️ Terminal")
    page = st.radio("View", PAGES, key="page")
    st.divider()
    st.markdown("**📡 Data source**")
    mode_label = st.selectbox("Mode", list(MODES), index=0, key="mode")
    MODE = MODES[mode_label]
    BROKER = st.selectbox("Broker API", ["None"] + list(B.PROVIDERS), index=0, key="broker")
    BROKER = "" if BROKER == "None" else BROKER
    creds = {}
    if BROKER:
        with st.expander(f"🔑 {BROKER} credentials", expanded=True):
            for fk, fl, secret in B.CRED_FIELDS[BROKER]:
                creds[fk] = st.text_input(fl, value=_secret(BROKER, fk), type="password" if secret else "default", key=f"c_{BROKER}_{fk}")
            st.caption("Kept in memory for this session only. Never written to disk or sent anywhere except the broker.")
    CFGT = (MODE, BROKER, tuple(sorted(creds.items())))
    CFG = H.Cfg(*CFGT)
    st.divider()
    universe_name = st.selectbox("Stock universe", ["Nifty 50", "Bank Nifty"])
    UNI = tuple(E.NIFTY50 if universe_name == "Nifty 50" else E.BANKNIFTY)
    max_n = st.slider("Max stocks", 8, len(UNI), min(len(UNI), 50), 1); UNI = UNI[:max_n]
    YEARS = {"3y": 3, "5y": 5, "10y": 10}[st.selectbox("History for back-tests", ["3y", "5y", "10y"], index=1)]
    st.markdown("**Trade model**")
    atr_sl = st.slider("Stop = ATR ×", 0.8, 3.0, 1.5, 0.1); rr = st.slider("Target = R multiple", 1.0, 4.0, 2.0, 0.5)
    hold = st.slider("Max hold (bars)", 3, 30, 10, 1); cost = st.slider("Cost per side %", 0.0, 0.3, 0.05, 0.01)
    PT = (atr_sl, rr, hold, cost, 3)
    if st.button("🔄 Refresh all data", width="stretch"):
        st.cache_data.clear(); st.rerun()
    st.caption("Educational analytics. Not investment advice.")


# ================================================================ cached loaders
@st.cache_data(ttl=900, show_spinner="Fetching daily data…")
def load_daily(syms: tuple, kind: str, cfgt: tuple, years: int):
    raw, src, msgs = H.get_daily(list(syms), kind, H.Cfg(*cfgt), years)
    return {k: E.prep(v) for k, v in raw.items()}, src, msgs

@st.cache_data(ttl=600, show_spinner="Fetching intraday data…")
def load_intraday(syms: tuple, kind: str, cfgt: tuple):
    return H.get_intraday(list(syms), kind, H.Cfg(*cfgt), "15m", 55)

@st.cache_data(ttl=900, show_spinner=False)
def load_global(demo: bool):
    return E.global_snapshot(demo)

@st.cache_data(ttl=1800, show_spinner=False)
def load_part(demo: bool):
    if not demo:
        df = E.fetch_participant_oi(5)
        if len(df):
            return df, False
    return E.demo_participant_oi(5), True

@st.cache_data(ttl=300, show_spinner=False)
def load_chain(sym: str, cfgt: tuple, spot):
    return H.get_option_chain(sym, H.Cfg(*cfgt), spot)

@st.cache_data(ttl=900, show_spinner="Running back-tests on every strategy…")
def load_backtests(uni: tuple, cfgt: tuple, years: int, p: tuple):
    data, _, _ = load_daily(uni, "stock", cfgt, years); bp = E.BTParams(*p)
    return {(k, side): E.run_strategy(data, k, side, bp) for k in E.STRATEGIES for side in ("LONG", "SHORT")}

@st.cache_data(ttl=900, show_spinner="Training walk-forward model…")
def load_predictions(uni: tuple, cfgt: tuple, years: int):
    idx_names = tuple(H.INDEX_YF)
    idata, isrc, imsg = load_daily(idx_names, "index", cfgt, years)
    sdata, ssrc, smsg = load_daily(uni, "stock", cfgt, years)
    mkt = idata.get("NIFTY 50")
    t_idx, ev_idx, m_idx = P.next_day_table(idata, mkt, "Index", l2=30.0)
    t_stk, ev_stk, m_stk = P.next_day_table(sdata, mkt, "Stock", l2=30.0)
    return {"idx": t_idx, "ev_idx": ev_idx, "m_idx": m_idx, "stk": t_stk, "ev_stk": ev_stk, "m_stk": m_stk,
            "src": {**{f"i:{k}": v for k, v in isrc.items()}, **{f"s:{k}": v for k, v in ssrc.items()}}, "msgs": imsg + smsg}

DEMO = MODE == "demo"

def get_context():
    gdf, gd = load_global(DEMO); part, pdm = load_part(DEMO)
    idata, isrc, imsg = load_daily(("NIFTY 50", "BANK NIFTY"), "index", CFGT, 5)
    nifty = idata["NIFTY 50"]; spot = float(nifty["Close"].iloc[-1])
    oc, osrc = load_chain("NIFTY", CFGT, spot); cs = E.chain_stats(oc)
    ctx = E.market_context(gdf, part, cs, nifty)
    srcs = {"NIFTY": isrc.get("NIFTY 50", "?"), "global": "DEMO" if gd else "Yahoo", "participant OI": "DEMO" if pdm else "NSE", "option chain": osrc}
    return dict(gdf=gdf, part=part, nifty=nifty, chain=cs, ctx=ctx, srcs=srcs, msgs=imsg)


now = datetime.now(IST)
st.markdown(f'<div class="hero"><h1>📈 NSE AI Smart Terminal</h1><p>Next-day forecast · Intraday desk · Hilega-Milega · Institutional flows · Option chain · Broker APIs (Dhan / Angel One / ICICI Breeze) · '
            f'Every signal back-tested · {now:%a %d %b %Y, %I:%M %p} IST</p></div>', unsafe_allow_html=True)


# ================================================================ PAGES
if page.startswith("🏠"):
    C = get_context(); data, src, msgs = load_daily(UNI, "stock", CFGT, YEARS); banner({**C["srcs"], **{f"s:{k}": v for k, v in src.items()}}, msgs + C["msgs"])
    ctx, n, cs = C["ctx"], C["nifty"].iloc[-1], C["chain"]
    c = st.columns(5)
    with c[0]: card("Market bias", ctx["label"], f"score {ctx['score']:+.1f} / 10", tone(ctx["score"]) if abs(ctx["score"]) >= 1 else AMBER)
    with c[1]: card("Nifty 50", f"{n['Close']:,.0f}", f"1D {n['Ret1']:+.2f}% · 5D {n['Ret5']:+.1f}% · 20D {n['Ret20']:+.1f}%", tone(n["Ret1"]))
    with c[2]: card("Nifty HM (D / W)", f"{n['HM_STATE']} / {n['WK_HM']}", f"ADX {n['ADX']:.0f} · RSI9 {n['RSI9']:.0f}", tone(1 if n["HM_STATE"] == "BULL" else -1 if n["HM_STATE"] == "BEAR" else 0))
    with c[3]: card("Option walls", f"{cs['sup']:,.0f} ↔ {cs['res']:,.0f}", f"PCR {cs['pcr']:.2f} · Max pain {cs['max_pain']:,.0f}", CYAN)
    vix = C["gdf"][C["gdf"]["Asset"] == "India VIX"]
    with c[4]: card("India VIX", f"{vix['Last'].iloc[0]:.2f}" if len(vix) else "n/a", f"{vix['Chg%'].iloc[0]:+.1f}% today" if len(vix) else "", AMBER)
    st.markdown("#### Why this bias")
    for t in ctx["notes"]: logic("• " + t)
    st.markdown("#### 🎯 Top live setups (back-test-weighted)")
    bt = load_backtests(UNI, CFGT, YEARS, PT); rows = []
    for s, d in data.items(): rows += E.confluence_row(s, d, bt, ctx)
    if rows:
        df = pd.DataFrame(rows); df["Grade"] = [E.grade(a, b, v) for a, b, v in zip(df["Score"], df["Prob%"], df["Verdict"])]
        top = df.sort_values("Score", ascending=False).head(10)[["Symbol", "Side", "Strategy", "Grade", "Score", "Prob%", "Entry~", "Stop", "Target", "Verdict", "Trades(n)"]]
        st.dataframe(style_tags(top, ["Side", "Grade", "Verdict"]).format({"Score": "{:.0f}", "Prob%": "{:.1f}", "Entry~": "{:,.2f}", "Stop": "{:,.2f}", "Target": "{:,.2f}"}), width="stretch", hide_index=True)
    else:
        st.info("No strategy is firing on the latest bar for this universe.")
    st.caption("Next: **Next-Day Predictions** for tomorrow's bull/bear lists, **Intraday Desk** for live-session factors, **Data Sources** to connect your broker.")

elif page.startswith("🔮"):
    st.subheader("Next trading day — Bull & Bear forecast (indices, sectors, stocks)")
    C = get_context(); PR = load_predictions(UNI, CFGT, YEARS); banner({**C["srcs"], **PR["src"]}, PR["msgs"])
    logic("<b>How it works:</b> a regularised logistic model predicts <i>P(next close &gt; today's close)</i> from trend, momentum, Hilega-Milega (D+W), ADX, MACD, volume, range position, 52-week distance, compression and Nifty context. "
          "It is validated <b>walk-forward</b> (trained only on the past, tested on the unseen future). Each live probability is shown next to the <b>measured out-of-sample hit-rate</b> of that confidence level, "
          "and a stock only earns grade A/B if that level has really worked (≥53% over ≥100 predictions). The market regime, option walls, FII flows and global cues are shown alongside, not blended in blindly.")
    ti, ts, evi, evs = PR["idx"], PR["stk"], PR["ev_idx"], PR["ev_stk"]
    # ---- market outlook
    if len(ti):
        nf = ti[ti["Symbol"] == "NIFTY 50"].iloc[0]; bn = ti[ti["Symbol"] == "BANK NIFTY"].iloc[0]; ctx, cs = C["ctx"], C["chain"]
        k = st.columns(5)
        with k[0]: card("Nifty 50 tomorrow", f"{nf['P(up) %']:.1f}% up", f"{nf['Direction']} · grade {nf['Grade']} · measured hit {nf['Measured hit-rate %']:.0f}%" if nf["Measured hit-rate %"] == nf["Measured hit-rate %"] else f"{nf['Direction']} · grade {nf['Grade']}", tone(nf["P(up) %"] - 50), border=tone(nf["P(up) %"] - 50))
        with k[1]: card("Bank Nifty tomorrow", f"{bn['P(up) %']:.1f}% up", f"{bn['Direction']} · grade {bn['Grade']}", tone(bn["P(up) %"] - 50), border=tone(bn["P(up) %"] - 50))
        with k[2]: card("Expected Nifty range", f"{nf['Exp. range low']:,.0f} – {nf['Exp. range high']:,.0f}", f"≈ ±{nf['Exp. move %']:.2f}% (0.6 ATR)", CYAN)
        with k[3]: card("Option walls", f"{cs['sup']:,.0f} ↔ {cs['res']:,.0f}", f"PCR {cs['pcr']:.2f} · max pain {cs['max_pain']:,.0f}", AMBER)
        with k[4]: card("Regime (global+flows+options)", ctx["label"], f"score {ctx['score']:+.1f}/10", tone(ctx["score"]))
        votes = [nf["P(up) %"] > 50, ctx["score"] > 0, cs["pcr"] > 1, C["nifty"].iloc[-1]["WK_HM"] == "BULL"]
        agree = sum(votes); tone_txt = "bullish" if agree >= 3 else "bearish" if agree <= 1 else "mixed"
        logic(f"<b>Tomorrow, in one line:</b> {agree}/4 independent components lean bullish (model, regime, PCR, weekly HM) → <b>{tone_txt}</b>. "
              f"Pivot {nf['Pivot']:,.0f} · S1 {nf['Support']:,.0f} · R1 {nf['Resistance']:,.0f}. Model drivers: {nf['Drivers']}. "
              f"Treat the probability as a tilt, not a guarantee — see the out-of-sample table below for how much each confidence level has actually delivered.")
    def eval_box(ev, label):
        with st.expander(f"📏 Out-of-sample evidence — {label}"):
            if ev.get("n", 0) < 200:
                st.info("Not enough history for a reliable walk-forward test."); return
            a, b, c3, d3 = st.columns(4)
            a.metric("OOS predictions", f"{ev['n']:,}"); b.metric("Direction accuracy", f"{ev['acc%']:.2f}%", f"{ev['acc%']-ev['naive%']:+.2f} vs 'always up' {ev['naive%']:.1f}%")
            c3.metric("Brier (lower better)", f"{ev['brier']:.4f}", f"{ev['brier']-ev['brier_base']:+.4f} vs base-rate"); d3.metric("Top-minus-bottom decile 1D", f"{ev['spread%']:+.3f}%")
            st.dataframe(ev["buckets"].style.format({"Hit-rate %": "{:.1f}", "Avg dir. move %": "{:+.3f}"}), hide_index=True, width="stretch")
            if ev["z_vs_naive"] < 2: st.warning(f"Overall accuracy is not statistically better than guessing (z = {ev['z_vs_naive']:.1f}). Daily direction is close to random — use these as weak tilts and rely on the grade/hit-rate columns.")
            else: st.success(f"Accuracy is statistically above a naive baseline (z = {ev['z_vs_naive']:.1f}). Edge is still small per trade; size positions accordingly.")
    def picks(df, title):
        if not len(df): st.info("No predictions."); return
        bull = df.sort_values("P(up) %", ascending=False).head(5); bear = df.sort_values("P(up) %").head(5)
        l, r = st.columns(2)
        with l:
            st.markdown(f"##### 🟢 Top bull — {title}")
            for _, x in bull.iterrows(): card(f"{x['Symbol']} · grade {x['Grade']} · {x['Close']:,.2f}", f"{x['P(up) %']:.1f}% up", f"{x['Playbook']}<br>Range {x['Exp. range low']:,.1f}–{x['Exp. range high']:,.1f} · S1 {x['Support']:,.1f} / R1 {x['Resistance']:,.1f}<br><i>{x['Drivers']}</i>", GREEN, GREEN); st.write("")
        with r:
            st.markdown(f"##### 🔴 Top bear — {title}")
            for _, x in bear.iterrows(): card(f"{x['Symbol']} · grade {x['Grade']} · {x['Close']:,.2f}", f"{100-x['P(up) %']:.1f}% down", f"{x['Playbook']}<br>Range {x['Exp. range low']:,.1f}–{x['Exp. range high']:,.1f} · S1 {x['Support']:,.1f} / R1 {x['Resistance']:,.1f}<br><i>{x['Drivers']}</i>", RED, RED); st.write("")
        cols = ["Rank", "Symbol", "Direction", "Grade", "P(up) %", "Measured hit-rate %", "Bucket n", "Close", "1D %", "Exp. move %", "Support", "Pivot", "Resistance", "HM D", "HM W", "ADX", "RSI14", "VolX", "Signals aligned", "Playbook", "Drivers"]
        st.dataframe(style_pm(style_tags(df[cols], ["Direction", "Grade", "HM D", "HM W"]), ["1D %"]).format({"P(up) %": "{:.1f}", "Measured hit-rate %": "{:.1f}", "Close": "{:,.2f}", "1D %": "{:+.2f}", "Exp. move %": "{:.2f}", "Support": "{:,.1f}", "Pivot": "{:,.1f}", "Resistance": "{:,.1f}", "ADX": "{:.0f}", "RSI14": "{:.0f}", "VolX": "{:.1f}"}), hide_index=True, width="stretch", height=380)
        st.download_button(f"⬇️ {title} CSV", df.to_csv(index=False), f"nextday_{title}.csv", "text/csv", key=f"dl_{title}")
    idata, _, _ = load_daily(tuple(H.INDEX_YF), "index", CFGT, YEARS); sdata, _, _ = load_daily(UNI, "stock", CFGT, YEARS)
    def add_aligned(df, dd):
        if len(df): df["Signals aligned"] = [sum(1 for sd in E.latest_signals(dd[s]).values() if (sd == "LONG") == (p > 50)) for s, p in zip(df["Symbol"], df["P(up) %"])]
        return df
    ti, ts = add_aligned(ti.copy(), idata), add_aligned(ts.copy(), sdata)
    tab1, tab2, tab3 = st.tabs(["🏛️ Indices", "🏭 Sectors", "📈 Stocks"])
    with tab1:
        picks(ti[ti["Symbol"].isin(["NIFTY 50", "BANK NIFTY", "SENSEX"])].reset_index(drop=True), "Indices"); eval_box(evi, "indices & sectors model")
    with tab2:
        picks(ti[~ti["Symbol"].isin(["NIFTY 50", "BANK NIFTY", "SENSEX"])].reset_index(drop=True), "Sectors")
    with tab3:
        picks(ts, "Stocks"); eval_box(evs, "stock model")
        if len(ts):
            st.markdown("#### Why a stock is ranked where it is")
            sel = st.selectbox("Stock", list(ts["Symbol"]), key="nd_sel"); d = sdata[sel]
            x = P.feature_frame(d, idata.get("NIFTY 50")).iloc[-1][P.FEATURES].values.astype(float)
            cn = P.contributions(PR["m_stk"], x); r_ = ts[ts["Symbol"] == sel].iloc[0]
            fig = go.Figure(go.Bar(x=[v for _, v in cn][::-1], y=[P.FEATURE_TXT[k] for k, _ in cn][::-1], orientation="h", marker_color=[GREEN if v > 0 else RED for _, v in cn][::-1]))
            fig.update_layout(title=f"{sel}: contribution to P(up) (standardised log-odds)"); st.plotly_chart(dark(fig, 480), width="stretch")
            logic(f"<b>{sel}</b>: P(up) {r_['P(up) %']:.1f}% → <b>{r_['Direction']}</b>, grade <b>{r_['Grade']}</b>. Daily/weekly HM {r_['HM D']}/{r_['HM W']}, ADX {r_['ADX']:.0f}, RSI14 {r_['RSI14']:.0f}. "
                  f"Plan: {r_['Playbook']}. Expected range {r_['Exp. range low']:,.1f}–{r_['Exp. range high']:,.1f}. {int(r_['Signals aligned'])} back-tested strategy signal(s) agree with the direction.")

elif page.startswith("⚡"):
    st.subheader("Intraday Desk — session factors, opening-range logic and next-session plan")
    grp = st.radio("Universe", ["Indices", "Stocks"], horizontal=True)
    syms = ("NIFTY 50", "BANK NIFTY", "SENSEX") if grp == "Indices" else UNI; kind = "index" if grp == "Indices" else "stock"
    d15, s15, m15 = load_intraday(syms, kind, CFGT); dd, _, _ = load_daily(syms, kind, CFGT, 2); banner({f"15m:{k}": v for k, v in s15.items()}, m15)
    if kind == "index":
        logic("Index volume is usually unavailable on free feeds, so index VWAP is <b>time-weighted</b> and relative volume is blank. Trade the index through its stocks/futures, or use a broker feed with futures volume.")
    rows, F = [], {}
    for s in syms:
        f = I.intraday_factors(d15[s], dd.get(s)) if s in d15 else None
        if f:
            F[s] = f
            rows.append({"Symbol": s, "Session": f["date"].strftime("%d %b"), "Bars": f["bars"], "Bias": f["bias"], "Score": f["score"], "Close": f["close"], "Day %": f["day_chg%"], "Gap %": f["gap%"],
                         "VWAP dist %": f["VWAP_dist%"], "ORB": f["ORB"], "vs PD range": f["vs_PDH"], "RVOL": f["rvol"], "RSI15": f["RSI15"], "Choppy": "YES" if f["choppy"] else "", "POC": f["POC"]})
    if not rows:
        st.info("Not enough intraday history.")
    else:
        df = pd.DataFrame(rows).sort_values("Score", ascending=False)
        last = max(f["date"] for f in F.values()); full = all(f["bars"] >= 24 for f in F.values())
        st.caption(f"Latest session in data: {last:%a %d %b %Y} — {'complete session (use for next-day plan)' if full else 'partial session (live)'}.")
        l, r = st.columns(2)
        with l:
            st.markdown("##### 🟢 Intraday bulls")
            for _, x in df.head(4).iterrows(): card(f"{x['Symbol']} · {x['Bias']}", f"{x['Score']:+.0f}", f"VWAP {x['VWAP dist %']:+.2f}% · {x['ORB']} · RVOL {x['RVOL']:.1f}x" if x["RVOL"] == x["RVOL"] else f"VWAP {x['VWAP dist %']:+.2f}% · {x['ORB']}", GREEN, GREEN); st.write("")
        with r:
            st.markdown("##### 🔴 Intraday bears")
            for _, x in df.tail(4).iloc[::-1].iterrows(): card(f"{x['Symbol']} · {x['Bias']}", f"{x['Score']:+.0f}", f"VWAP {x['VWAP dist %']:+.2f}% · {x['ORB']} · RVOL {x['RVOL']:.1f}x" if x["RVOL"] == x["RVOL"] else f"VWAP {x['VWAP dist %']:+.2f}% · {x['ORB']}", RED, RED); st.write("")
        st.dataframe(style_pm(style_tags(df, ["Bias"]), ["Day %", "Gap %", "VWAP dist %", "Score"]).format({"Score": "{:+.0f}", "Close": "{:,.2f}", "Day %": "{:+.2f}", "Gap %": "{:+.2f}", "VWAP dist %": "{:+.2f}", "RVOL": "{:.1f}", "RSI15": "{:.0f}", "POC": "{:,.1f}"}), hide_index=True, width="stretch", height=380)
        st.markdown("#### Session inspector")
        sel = st.selectbox("Symbol", list(F), key="id_sel"); f = F[sel]; g = f["frame"]
        fig = go.Figure(go.Candlestick(x=g.index, open=g["Open"], high=g["High"], low=g["Low"], close=g["Close"], name=sel))
        fig.add_trace(go.Scatter(x=g.index, y=g["VWAP"], name="VWAP", line=dict(color=AMBER, width=2)))
        for lvl, nm, col in [(f["OR_high"], "OR high", GREEN), (f["OR_low"], "OR low", RED), (f["PDH"], "PDH", VIOLET), (f["PDL"], "PDL", VIOLET), (f["POC"], "POC", CYAN)]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=col, annotation_text=f"{nm} {lvl:,.1f}", annotation_position="right")
        fig.add_hrect(y0=min(f["CPR"]), y1=max(f["CPR"]), fillcolor=CYAN, opacity=.08, line_width=0, annotation_text="CPR")
        fig.update_layout(xaxis_rangeslider_visible=False, title=f"{sel} — {f['date']:%d %b %Y} 15-minute"); st.plotly_chart(dark(fig, 520), width="stretch")
        a, b = st.columns([1, 1])
        with a:
            st.dataframe(pd.DataFrame([{"Factor": t, "Points": v} for v, t in f["why"]]).style.format({"Points": "{:+.1f}"}).map(lambda v: f"color:{GREEN}" if isinstance(v, (int, float)) and v > 0 else f"color:{RED}" if isinstance(v, (int, float)) and v < 0 else "", subset=["Points"]), hide_index=True, width="stretch")
        with b:
            logic(f"<b>Intraday read: {f['bias']} ({f['score']:+.0f}).</b> Price {f['close']:,.2f} vs VWAP {f['VWAP']:,.2f}; opening range {f['OR_low']:,.1f}–{f['OR_high']:,.1f} ({f['ORB'].lower()}); gap {f['gap%']:+.2f}%; "
                  f"first-hour {f['first_hour%']:+.2f}%; VWAP crossed {f['vwap_cross']}×{' → choppy, favour mean-reversion/ no breakout trades' if f['choppy'] else ''}. CPR width {f['CPR_w%']:.2f}% ({'narrow → trend-day odds' if f['CPR_w%'] < 0.25 else 'normal/wide → range-day odds'}).")
        st.markdown("#### 🧪 Intraday back-test (this feed's history)")
        n_sess = len(I.sessions(next(iter(d15.values()))))
        logic(f"Rules are fixed, entries at bar close, stop at the opposite side of the range (or VWAP-based), target {1.5}R, flat by close, costs {0.03}% per side. Sample: <b>{n_sess} sessions</b>. "
              f"Free feeds give ≤60 days, so conclusions are weak — connect a broker for years of 15-minute history.")
        bt15 = I.intraday_backtest({s: d15[s] for s in syms if s in d15}); rows = []
        for var, name in I.INTRADAY_VARIANTS.items():
            r_ = bt15[var]
            for sd in ("LONG", "SHORT", "ALL"):
                a_ = r_[sd]
                rows.append({"Strategy": name, "Side": sd, "Trades": a_["n"], "Win %": a_["win%"], "Avg R": a_["avgR"], "PF": a_["PF"], "Max DD %": a_["maxDD%"], "OOS trades": r_["oos"]["n"] if sd == "ALL" else np.nan, "OOS Avg R": r_["oos"]["avgR"] if sd == "ALL" else np.nan})
        st.dataframe(style_pm(pd.DataFrame(rows), ["Avg R", "OOS Avg R"]).format({"Win %": "{:.1f}", "Avg R": "{:+.2f}", "PF": "{:.2f}", "Max DD %": "{:.1f}", "OOS Avg R": "{:+.2f}"}), hide_index=True, width="stretch")
        st.markdown("#### 📅 Next-session plan (from the latest daily bar)")
        pk = st.selectbox("Plan for", list(syms), key="pl_sel")
        if pk in dd:
            pl = I.next_session_plan(dd[pk]); pv = pl["pivots"]
            k = st.columns(4)
            with k[0]: card("Long above", f"{pl['long_trigger']:,.1f}", f"SL {pl['long_sl']:,.1f} · T1 {pl['long_t1']:,.1f} · T2 {pl['long_t2']:,.1f}", GREEN)
            with k[1]: card("Short below", f"{pl['short_trigger']:,.1f}", f"SL {pl['short_sl']:,.1f} · T1 {pl['short_t1']:,.1f} · T2 {pl['short_t2']:,.1f}", RED)
            with k[2]: card("Expected range", f"{pl['exp_range'][0]:,.1f} – {pl['exp_range'][1]:,.1f}", f"ATR {pl['atr%']:.2f}%", CYAN)
            with k[3]: card("Pivot / CPR", f"{pv['PP']:,.1f}", f"S1 {pv['S1']:,.1f} · R1 {pv['R1']:,.1f} · CPR {pv['BC']:,.1f}-{pv['TC']:,.1f}", VIOLET)
            logic("<b>Playbook:</b> take the long only if price trades above yesterday's high and holds above VWAP after the first 30 minutes; short only below yesterday's low under VWAP. "
                  "A gap beyond the trigger → wait for the first 15-minute candle to close before entering (avoid chasing). Flags: " + (", ".join(pl["tags"]) if pl["tags"] else "none") + ".")

elif page.startswith("🎯"):
    C = get_context(); data, src, msgs = load_daily(UNI, "stock", CFGT, YEARS); banner({f"s:{k}": v for k, v in src.items()}, msgs)
    bt = load_backtests(UNI, CFGT, YEARS, PT); ctx = C["ctx"]
    st.subheader("High-Probability Setups — confluence of every module")
    logic("<b>Probability =</b> the strategy's own historical win-rate (shrunk toward a random-entry baseline) tilted by live confluence — daily + weekly HM, 200-DMA, ADX, MACD, volume, 52W position, option walls, FII, global — capped 25–80%.")
    f1, f2, f3, f4 = st.columns(4)
    side_f = f1.selectbox("Side", ["All", "LONG", "SHORT"]); min_n = f2.slider("Min historical trades", 20, 500, 60, 10)
    min_s = f3.slider("Min score", 0, 90, 45, 5); only_edge = f4.checkbox("Only strategies with edge", True)
    rows = []
    for s, d in data.items(): rows += E.confluence_row(s, d, bt, ctx, None if side_f == "All" else side_f)
    if not rows: st.info("Nothing firing on the latest bar.")
    else:
        df = pd.DataFrame(rows); df["Grade"] = [E.grade(a, b, v) for a, b, v in zip(df["Score"], df["Prob%"], df["Verdict"])]
        df = df[(df["Trades(n)"] >= min_n) & (df["Score"] >= min_s)]
        if only_edge: df = df[df["Verdict"].isin(["ROBUST EDGE", "PROBABLE EDGE", "MARGINAL"])]
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        k = st.columns(4)
        with k[0]: card("Setups shown", len(df), f"{(df['Side']=='LONG').sum()} long · {(df['Side']=='SHORT').sum()} short")
        with k[1]: card("A / A+ grade", int(df["Grade"].isin(["A", "A+"]).sum()), "score≥62 & prob≥54%", GREEN)
        with k[2]: card("Avg probability", f"{df['Prob%'].mean():.1f}%" if len(df) else "–", "model estimate", CYAN)
        with k[3]: card("Market regime", ctx["label"], f"{ctx['score']:+.1f}", tone(ctx["score"]))
        cols = ["Symbol", "Side", "Strategy", "Grade", "Score", "Prob%", "Entry~", "Stop", "Target", "Risk%", "Hist Win%", "Edge pp", "Hist AvgR", "PF", "Trades(n)", "OOS AvgR", "Verdict", "HM D", "HM W", "ADX", "VolX"]
        st.dataframe(style_tags(df[cols], ["Side", "Grade", "Verdict", "HM D", "HM W"]).format({"Score": "{:.0f}", "Prob%": "{:.1f}", "Entry~": "{:,.2f}", "Stop": "{:,.2f}", "Target": "{:,.2f}", "Risk%": "{:.2f}", "Hist Win%": "{:.1f}", "Edge pp": "{:+.1f}", "Hist AvgR": "{:+.2f}", "PF": "{:.2f}", "OOS AvgR": "{:+.2f}", "ADX": "{:.0f}", "VolX": "{:.1f}"}), width="stretch", hide_index=True, height=420)
        st.download_button("⬇️ Download setups CSV", df.to_csv(index=False), "setups.csv", "text/csv")
        if len(df):
            labels = [f"{r.Symbol} · {r.Side} · {r.Strategy}" for r in df.itertuples()]; pick = st.selectbox("Inspect setup", labels); r = df.iloc[labels.index(pick)]
            logic(f"<b>Why considered:</b> {E.STRATEGIES[r['Key']].why}<br><b>Rule:</b> {E.STRATEGIES[r['Key']].logic}<br><b>Live confluence:</b> {r['Why']}")
            logic(f"<b>Evidence:</b> {int(r['Trades(n)'])} historical trades, win-rate {r['Hist Win%']:.1f}% (edge {r['Edge pp']:+.1f}pp vs random entry), avg {r['Hist AvgR']:+.2f}R, PF {r['PF']:.2f}, out-of-sample {r['OOS AvgR']:+.2f}R → <b>{r['Verdict']}</b>. Combined probability <b>{r['Prob%']:.1f}%</b>.")
            d = data[r["Symbol"]].tail(160); fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=r["Symbol"]))
            for lvl, nm, col in [(r["Entry~"], "Entry", CYAN), (r["Stop"], "Stop", RED), (r["Target"], "Target", GREEN)]: fig.add_hline(y=lvl, line_dash="dash", line_color=col, annotation_text=f"{nm} {lvl:,.2f}")
            fig.add_trace(go.Scatter(x=d.index, y=d["EMA20"], name="EMA20", line=dict(color=AMBER, width=1))); fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"], name="SMA200", line=dict(color=VIOLET, width=1)))
            fig.update_layout(xaxis_rangeslider_visible=False); st.plotly_chart(dark(fig, 480), width="stretch")

elif page.startswith("🧪"):
    data, src, msgs = load_daily(UNI, "stock", CFGT, YEARS); banner({f"s:{k}": v for k, v in src.items()}, msgs)
    bt = load_backtests(UNI, CFGT, YEARS, PT)
    st.subheader("Backtest Lab — does each script's logic actually have an edge?")
    logic("Entry at the <b>next bar's open</b>, ATR stop, R-multiple target, costs both sides, stop assumed first if both levels trade in one bar. A <b>random-entry baseline</b> uses the identical bracket. 60/40 in-time split: edge must survive <b>out-of-sample</b>.")
    rows = []
    for (k, side), r in bt.items():
        a, o, v = r["all"], r["oos"], E.verdict(r)
        rows.append({"Strategy": E.STRATEGIES[k].name, "Key": k, "Side": side, "Trades": a["n"], "Win%": a["win%"], "Base Win%": r["base_wr"] * 100 if r["base_wr"] == r["base_wr"] else np.nan, "Edge pp": a["edge_pp"], "Avg R": a["avgR"], "PF": a["PF"], "Max DD %": a["maxDD%"], "Prob% (shrunk)": a["prob%"], "Wilson LB%": a["wilson%"], "OOS n": o["n"], "OOS Avg R": o["avgR"], "OOS PF": o["PF"], "Verdict": v[0]})
    tab = pd.DataFrame(rows).sort_values("Avg R", ascending=False)
    st.dataframe(style_tags(tab.drop(columns="Key"), ["Side", "Verdict"]).format({"Win%": "{:.1f}", "Base Win%": "{:.1f}", "Edge pp": "{:+.1f}", "Avg R": "{:+.2f}", "PF": "{:.2f}", "Max DD %": "{:.1f}", "Prob% (shrunk)": "{:.1f}", "Wilson LB%": "{:.1f}", "OOS Avg R": "{:+.2f}", "OOS PF": "{:.2f}"}), hide_index=True, width="stretch", height=520)
    t2 = tab[tab["Trades"] >= 30]; xs = [f"{a} {b}" for a, b in zip(t2["Key"], t2["Side"])]
    fig = go.Figure([go.Bar(x=xs, y=t2["Avg R"], name="All-period Avg R", marker_color=CYAN), go.Bar(x=xs, y=t2["OOS Avg R"], name="Out-of-sample Avg R", marker_color=AMBER)])
    fig.update_layout(barmode="group", title="Edge per trade (R) — all period vs out-of-sample"); st.plotly_chart(dark(fig, 380), width="stretch")
    st.markdown("#### Deep-dive")
    c1, c2 = st.columns(2); key = c1.selectbox("Strategy", list(E.STRATEGIES), format_func=lambda k: E.STRATEGIES[k].name); side = c2.radio("Side", ["LONG", "SHORT"], horizontal=True)
    r = bt[(key, side)]; v = E.verdict(r); stg = E.STRATEGIES[key]
    logic(f"<b>{stg.name}</b> <i>(from {stg.source})</i><br><b>Rule:</b> {stg.logic}<br><b>Why it might work:</b> {stg.why}<br><b>Verdict:</b> {v[0]} — {v[1]}")
    if len(r["trades"]):
        a, o = r["all"], r["oos"]; k = st.columns(5)
        with k[0]: card("Trades", a["n"], f"avg hold {a['avg_bars']:.1f} bars")
        with k[1]: card("Win rate", f"{a['win%']:.1f}%", f"baseline {r['base_wr']*100:.1f}% · edge {a['edge_pp']:+.1f}pp", tone(a["edge_pp"]))
        with k[2]: card("Avg R / trade", f"{a['avgR']:+.2f}", f"PF {a['PF']:.2f}", tone(a["avgR"]))
        with k[3]: card("Calibrated prob", f"{a['prob%']:.1f}%", f"Wilson lower bound {a['wilson%']:.1f}%", CYAN)
        with k[4]: card("Out-of-sample", f"{o['avgR']:+.2f} R", f"{o['n']} trades · PF {o['PF']:.2f}", tone(o["avgR"]))
        eq = r["equity"]; fig = go.Figure(go.Scatter(x=eq["Exit"], y=eq["Cum"], fill="tozeroy", line=dict(color=GREEN))); fig.add_vline(x=r["cut"], line_dash="dot", line_color=AMBER, annotation_text="out-of-sample →")
        fig.update_layout(title="Cumulative net % (sum of trade returns, all stocks pooled)"); st.plotly_chart(dark(fig, 340), width="stretch")
        tr = r["trades"]; c1, c2 = st.columns(2)
        with c1:
            h = go.Figure(go.Histogram(x=tr["R"].clip(-3, 5), nbinsx=40, marker_color=VIOLET)); h.update_layout(title="Distribution of R-multiples"); st.plotly_chart(dark(h, 300), width="stretch")
        with c2:
            by = tr.assign(Y=tr["Signal"].dt.year).groupby("Y")["R"].agg(["mean", "count"]).reset_index()
            b = go.Figure(go.Bar(x=by["Y"], y=by["mean"], marker_color=[GREEN if x > 0 else RED for x in by["mean"]], text=by["count"])); b.update_layout(title="Avg R by year (label = trades)"); st.plotly_chart(dark(b, 300), width="stretch")
        d0 = next(iter(data.values())); sig = (stg.long_fn if side == "LONG" else stg.short_fn)(d0)
        st.caption("Forward-return study on the first stock (no stop/target, entry at signal close):")
        st.dataframe(pd.DataFrame(E.forward_stats(d0, sig, side)).T.rename_axis("bars ahead").style.format("{:.2f}"), width="stretch")
        st.download_button("⬇️ Download all trades CSV", tr.to_csv(index=False), f"{key}_{side}_trades.csv", "text/csv")
    else: st.info("No trades generated.")

elif page.startswith("🌍"):
    C = get_context(); banner(C["srcs"], C["msgs"]); st.subheader("Global cues & cross-asset macro"); g = C["gdf"]; cols = st.columns(4)
    for i, r in g.iterrows():
        with cols[i % 4]:
            inv = r["Asset"] in ("India VIX", "Dollar Index", "USD/INR", "Brent", "US 10Y")
            card(r["Asset"], f"{r['Last']:,.2f}", f"{r['Chg%']:+.2f}% · {r['Impact on India']}", tone(-r["Chg%"] if inv else r["Chg%"])); st.write("")
    logic("<b>Logic:</b> US/Asia futures up = positive for India; India VIX, dollar, USD/INR, crude and US 10Y rising = headwinds. A move below ±0.4% is treated as noise. These feed the −10…+10 regime score.")
    assets = {"Nifty 50": "^NSEI", "S&P 500": "^GSPC", "Gold": "GC=F", "Crude": "CL=F", "USD/INR": "INR=X", "Bitcoin": "BTC-USD", "India VIX": "^INDIAVIX"}
    @st.cache_data(ttl=1800, show_spinner=False)
    def corr(demo):
        return pd.DataFrame({n: E.fetch_history(t, "1y", demo=demo)[0]["Close"] for n, t in assets.items()}).pct_change().dropna(how="all").corr().round(2)
    cm = corr(DEMO); fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, zmin=-1, zmax=1, colorscale="RdYlGn", text=cm.values, texttemplate="%{text}"))
    st.markdown("#### Correlation matrix (daily returns, 1y)"); st.plotly_chart(dark(fig, 440), width="stretch")

elif page.startswith("📊"):
    st.subheader("Index technicals — multi-timeframe")
    idata, src, msgs = load_daily(tuple(H.INDEX_YF), "index", CFGT, YEARS); banner({k: v for k, v in src.items()}, msgs)
    name = st.selectbox("Index", list(idata)); d = idata[name]; n = d.iloc[-1]; nxt = E.pivots(n["High"], n["Low"], n["Close"]); k = st.columns(6)
    with k[0]: card("Close", f"{n['Close']:,.1f}", f"{n['Ret1']:+.2f}%", tone(n["Ret1"]))
    with k[1]: card("HM Daily", n["HM_STATE"], f"streak {int(n['HM_STREAK'])}", tone(1 if n["HM_STATE"] == "BULL" else -1 if n["HM_STATE"] == "BEAR" else 0))
    with k[2]: card("HM Weekly", n["WK_HM"], "last completed week", tone(1 if n["WK_HM"] == "BULL" else -1 if n["WK_HM"] == "BEAR" else 0))
    with k[3]: card("ADX / RSI14", f"{n['ADX']:.0f} / {n['RSI14']:.0f}", "trend strength / momentum", CYAN)
    with k[4]: card("ATR", f"{n['ATR']:.0f}", f"{n['ATR%']:.2f}% daily range", AMBER)
    with k[5]: card("Next-day pivot", f"{nxt['PP']:,.0f}", f"S1 {nxt['S1']:,.0f} · R1 {nxt['R1']:,.0f}", VIOLET)
    t = d.tail(180); fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.6, .2, .2], vertical_spacing=.03)
    fig.add_trace(go.Candlestick(x=t.index, open=t["Open"], high=t["High"], low=t["Low"], close=t["Close"], name="Price"), 1, 1)
    for col, c_ in [("EMA20", AMBER), ("SMA50_", CYAN), ("SMA200", VIOLET)]: fig.add_trace(go.Scatter(x=t.index, y=t[col], name=col.strip("_"), line=dict(width=1, color=c_)), 1, 1)
    bu = t[t["HM_BULL"] & (t["HM_STREAK"] == 1)]; be = t[t["HM_BEAR"] & (t["HM_STREAK"] == 1)]
    fig.add_trace(go.Scatter(x=bu.index, y=bu["Low"] * .995, mode="markers", marker=dict(symbol="triangle-up", size=10, color=GREEN), name="HM bull start"), 1, 1)
    fig.add_trace(go.Scatter(x=be.index, y=be["High"] * 1.005, mode="markers", marker=dict(symbol="triangle-down", size=10, color=RED), name="HM bear start"), 1, 1)
    for col, c_ in [("RSI9", CYAN), ("EMA3", AMBER), ("WMA21", VIOLET)]: fig.add_trace(go.Scatter(x=t.index, y=t[col], name=col, line=dict(width=1.2, color=c_)), 2, 1)
    fig.add_hline(y=50, row=2, col=1, line_dash="dot", line_color=MUTED)
    fig.add_trace(go.Bar(x=t.index, y=t["MACD_H"], name="MACD hist", marker_color=[GREEN if v > 0 else RED for v in t["MACD_H"]]), 3, 1)
    fig.update_layout(xaxis_rangeslider_visible=False); st.plotly_chart(dark(fig, 760), width="stretch")
    mt = pd.DataFrame([{"Timeframe": "Weekly HM", "State": n["WK_HM"]}, {"Timeframe": "Daily HM", "State": n["HM_STATE"]}, {"Timeframe": "Price vs EMA20", "State": "BULL" if n["Close"] > n["EMA20"] else "BEAR"},
                       {"Timeframe": "Price vs SMA200", "State": "BULL" if n["Close"] > n["SMA200"] else "BEAR"}, {"Timeframe": "MACD", "State": "BULL" if n["MACD_H"] > 0 else "BEAR"}])
    c1, c2 = st.columns(2)
    with c1: st.dataframe(style_tags(mt, ["State"]), hide_index=True, width="stretch")
    with c2: st.dataframe(pd.DataFrame({"Level": ["R2", "R1", "TC", "Pivot", "BC", "S1", "S2"], "Value": [nxt["R2"], nxt["R1"], nxt["TC"], nxt["PP"], nxt["BC"], nxt["S1"], nxt["S2"]]}).style.format({"Value": "{:,.1f}"}), hide_index=True, width="stretch")
    logic("Hilega-Milega bull = RSI9>50, its 3-EMA ≥ 50, its 21-WMA below the 3-EMA, price ≥ SMA50. Daily + weekly agreeing is what the Confluence strategy requires.")

elif page.startswith("📶"):
    data, src, msgs = load_daily(UNI, "stock", CFGT, YEARS); idata, isrc, imsg = load_daily(tuple(H.INDEX_YF), "index", CFGT, YEARS); banner({**{f"s:{k}": v for k, v in src.items()}, **isrc}, msgs + imsg)
    st.subheader("Market breadth & sector rotation"); last = pd.DataFrame({s: d.iloc[-1] for s, d in data.items()}).T; adv, dec = int((last["Ret1"] > 0).sum()), int((last["Ret1"] < 0).sum()); k = st.columns(6)
    with k[0]: card("Advance / Decline", f"{adv} / {dec}", f"A/D {adv/max(dec,1):.2f}", tone(adv - dec))
    with k[1]: card("> SMA20", f"{(last['Close']>last['SMA20']).mean()*100:.0f}%", "short-term", CYAN)
    with k[2]: card("> SMA50", f"{(last['Close']>last['SMA50_']).mean()*100:.0f}%", "medium-term", CYAN)
    with k[3]: card("> SMA200", f"{(last['Close']>last['SMA200']).mean()*100:.0f}%", "long-term", CYAN)
    with k[4]: card("HM bull / bear", f"{int((last['HM_STATE']=='BULL').sum())} / {int((last['HM_STATE']=='BEAR').sum())}", "daily", GREEN)
    with k[5]: card("20D highs / lows", f"{int((last['Close']>last['Hi20']).sum())} / {int((last['Close']<last['Lo20']).sum())}", "breakouts", AMBER)
    hist = pd.DataFrame({s: (d["Close"] > d["SMA50_"]) for s, d in data.items()}).astype(float).mean(axis=1).dropna().tail(250) * 100
    fig = go.Figure(go.Scatter(x=hist.index, y=hist, fill="tozeroy", line=dict(color=CYAN))); fig.add_hline(y=50, line_dash="dot", line_color=MUTED); fig.update_layout(title="% of stocks above 50-DMA"); st.plotly_chart(dark(fig, 300), width="stretch")
    st.markdown("#### Sector rotation (relative strength vs Nifty)"); nf = idata["NIFTY 50"]; rows = []
    for nm, d in idata.items():
        if nm in ("NIFTY 50", "SENSEX", "BANK NIFTY"): continue
        x = d.iloc[-1]; rs20, rs5 = x["Ret20"] - nf.iloc[-1]["Ret20"], x["Ret5"] - nf.iloc[-1]["Ret5"]; q = ("LEADING" if rs5 > 0 else "WEAKENING") if rs20 > 0 else ("IMPROVING" if rs5 > 0 else "LAGGING")
        rows.append({"Index": nm, "1D %": x["Ret1"], "5D %": x["Ret5"], "20D %": x["Ret20"], "RS20 pp": rs20, "RS5 pp": rs5, "Quadrant": q, "HM D": x["HM_STATE"], "HM W": x["WK_HM"]})
    sec = pd.DataFrame(rows).sort_values("20D %", ascending=False)
    fig = go.Figure(go.Scatter(x=sec["RS20 pp"], y=sec["RS5 pp"], mode="markers+text", text=sec["Index"].str.replace("NIFTY ", ""), textposition="top center", marker=dict(size=14, color=[GREEN if q == "LEADING" else CYAN if q == "IMPROVING" else AMBER if q == "WEAKENING" else RED for q in sec["Quadrant"]])))
    fig.add_hline(y=0, line_color=MUTED); fig.add_vline(x=0, line_color=MUTED); fig.update_layout(xaxis_title="20D relative strength (pp)", yaxis_title="5D relative strength (pp)"); st.plotly_chart(dark(fig, 440), width="stretch")
    st.dataframe(style_pm(style_tags(sec, ["Quadrant", "HM D", "HM W"]), ["1D %", "5D %", "20D %", "RS20 pp", "RS5 pp"]).format({c: "{:+.2f}" for c in ["1D %", "5D %", "20D %", "RS20 pp", "RS5 pp"]}), hide_index=True, width="stretch")

elif page.startswith("🔁"):
    data, src, msgs = load_daily(UNI, "stock", CFGT, YEARS); banner({f"s:{k}": v for k, v in src.items()}, msgs); st.subheader("Hilega-Milega screener")
    logic("Bull: RSI9>50 · EMA3(RSI)≥50 · WMA21(RSI)≤EMA3 · Close≥SMA50. Bear: mirror. <b>Streak</b> = bars in state (fresh = ≤2). Tags mirror the original script. <b>Del %</b> appears when NSE bhavcopy is the source.")
    rows = []
    for s, d in data.items():
        x = d.iloc[-1]
        if x["HM_STATE"] == "NEUTRAL": continue
        L = x["HM_STATE"] == "BULL"
        rows.append({"Symbol": s, "HM D": x["HM_STATE"], "HM W": x["WK_HM"], "Streak": int(x["HM_STREAK"]), "Close": x["Close"], "1D %": x["Ret1"], "5D %": x["Ret5"], "RSI9": x["RSI9"], "ADX": x["ADX"], "VolX": x["VolX"],
                     "Del %": x["DelPct"] if "DelPct" in d.columns else np.nan, "5EMA Tag": ("Hilega5EMABull" if x["EMA5"] > x["Low"] else "") if L else ("Milega5EMABear" if x["EMA5"] < x["High"] else ""),
                     "Weekly BO": "5dayweekBullBO" if x["WK_BO_UP"] else "5dayweekBearBO" if x["WK_BO_DN"] else "", "Aligned D+W": "YES" if x["HM_STATE"] == x["WK_HM"] else "", "Stop (1.5ATR)": x["Close"] - 1.5 * x["ATR"] if L else x["Close"] + 1.5 * x["ATR"]})
    if rows:
        df = pd.DataFrame(rows); c1, c2 = st.columns(2); sd = c1.radio("Show", ["Both", "BULL", "BEAR"], horizontal=True); fr = c2.checkbox("Fresh only (streak ≤2)", False)
        if sd != "Both": df = df[df["HM D"] == sd]
        if fr: df = df[df["Streak"] <= 2]
        df = df.sort_values(["Aligned D+W", "ADX"], ascending=False)
        st.dataframe(style_pm(style_tags(df, ["HM D", "HM W"]), ["1D %", "5D %"]).format({"Close": "{:,.2f}", "1D %": "{:+.2f}", "5D %": "{:+.2f}", "RSI9": "{:.0f}", "ADX": "{:.0f}", "VolX": "{:.1f}", "Del %": "{:.0f}", "Stop (1.5ATR)": "{:,.2f}"}), hide_index=True, width="stretch", height=520)
        st.download_button("⬇️ CSV", df.to_csv(index=False), "hm_screener.csv", "text/csv")
    else: st.info("No stock currently in a Hilega-Milega state.")

elif page.startswith("🕯️"):
    data, src, msgs = load_daily(UNI, "stock", CFGT, YEARS); banner({f"s:{k}": v for k, v in src.items()}, msgs); st.subheader("Heikin-Ashi & pattern screener (all strategies)")
    keys = st.multiselect("Strategies", list(E.STRATEGIES), default=["HA_REBOUND", "EMA_DIP", "NR7", "VOL_BURST", "WEEK_BO", "BRK20"], format_func=lambda k: E.STRATEGIES[k].name); rows = []
    for s, d in data.items():
        x = d.iloc[-1]
        for k_, side in E.latest_signals(d).items():
            if k_ in keys: rows.append({"Symbol": s, "Strategy": E.STRATEGIES[k_].name, "Side": side, "Close": x["Close"], "RSI14": x["RSI14"], "ADX": x["ADX"], "VolX": x["VolX"], "VWAP dist %": x["VWAPd%"], "HA": "Green" if x["HA_Green"] else "Red", "NR7": "YES" if x["NR7"] else "", "Inside bar": "YES" if x["INSIDE"] else ""})
    if rows: st.dataframe(style_tags(pd.DataFrame(rows), ["Side"]).format({"Close": "{:,.2f}", "RSI14": "{:.0f}", "ADX": "{:.0f}", "VolX": "{:.1f}", "VWAP dist %": "{:+.2f}"}), hide_index=True, width="stretch", height=480)
    else: st.info("No matches on the latest bar.")

elif page.startswith("🏦"):
    part, pdm = load_part(DEMO); banner({"participant OI": "DEMO" if pdm else "NSE"}); st.subheader("Institutional flows — FII / DII / Pro / Client (index derivatives)")
    last = part[part["Date"] == part["Date"].max()].set_index("Participant"); k = st.columns(4)
    for i, w in enumerate(["FII", "DII", "PRO", "CLIENT"]):
        if w in last.index:
            with k[i]: card(f"{w} net futures", f"{last.loc[w,'Net Futures']:+,.0f}", f"{last.loc[w,'Long %']:.1f}% long", tone(last.loc[w, "Net Futures"]))
    pv = part.pivot(index="Date", columns="Participant", values="Net Futures"); fig = go.Figure([go.Bar(x=pv.index, y=pv[c], name=c) for c in pv.columns]); fig.update_layout(barmode="group", title="Net index-futures position by participant"); st.plotly_chart(dark(fig, 340), width="stretch")
    f = part[part["Participant"] == "FII"]; fig = go.Figure([go.Bar(x=f["Date"], y=f["Net Calls"], name="FII net calls", marker_color=GREEN), go.Bar(x=f["Date"], y=f["Net Puts"], name="FII net puts", marker_color=RED)]); fig.update_layout(barmode="group", title="FII net option positions"); st.plotly_chart(dark(fig, 300), width="stretch")
    if {"FII", "CLIENT", "DII"} <= set(last.index):
        fii, cl, di = last.loc["FII"], last.loc["CLIENT"], last.loc["DII"]
        if fii["Net Futures"] < -50000 and cl["Net Futures"] > 50000: logic("🚨 <b>Retail trap:</b> clients net-long while FIIs are heavily net-short.")
        elif fii["Net Futures"] > 50000 and cl["Net Futures"] < -50000: logic("🚀 <b>Squeeze setup:</b> clients net-short while FIIs are heavily net-long.")
        if di["Net Futures"] > 25000 and fii["Net Futures"] < -50000: logic("🛡️ <b>DII absorption:</b> DIIs buying what FIIs sell — consolidation more likely than collapse.")
        if fii["Long %"] < 40: logic(f"📉 FII structurally bearish: only {fii['Long %']:.1f}% of index futures are long.")
        elif fii["Long %"] > 60: logic(f"📈 FII structurally bullish: {fii['Long %']:.1f}% of index futures are long.")
    st.dataframe(style_pm(part.sort_values("Date", ascending=False), ["Net Futures", "Net Calls", "Net Puts"]), hide_index=True, width="stretch")
    st.caption("Source: NSE fao_participant_oi archive (free, official). Thresholds are heuristics from Market_Decoder.py, not statistically tuned.")

elif page.startswith("🧩"):
    sym = st.radio("Index", ["NIFTY", "BANKNIFTY"], horizontal=True); idata, _, _ = load_daily(("NIFTY 50", "BANK NIFTY"), "index", CFGT, 2)
    spot = float(idata["NIFTY 50" if sym == "NIFTY" else "BANK NIFTY"]["Close"].iloc[-1]); oc, osrc = load_chain(sym, CFGT, spot); banner({"option chain": osrc}); cs = E.chain_stats(oc); k = st.columns(5)
    with k[0]: card("PCR (OI)", f"{cs['pcr']:.2f}", ">1.3 supportive · <0.7 bearish", GREEN if cs["pcr"] > 1 else RED)
    with k[1]: card("Resistance (max CE OI)", f"{cs['res']:,.0f}", f"then {cs['top_ce'][1]:,.0f}, {cs['top_ce'][2]:,.0f}", RED)
    with k[2]: card("Support (max PE OI)", f"{cs['sup']:,.0f}", f"then {cs['top_pe'][1]:,.0f}, {cs['top_pe'][2]:,.0f}", GREEN)
    with k[3]: card("Max pain", f"{cs['max_pain']:,.0f}", "price magnet into expiry", AMBER)
    with k[4]: card("Expiry", str(cs["expiry"]), f"spot {cs['spot']:,.0f}" if cs["spot"] else "", CYAN)
    ks = sorted(oc["ce"]); fig = go.Figure([go.Bar(x=ks, y=[oc["ce"][s][0] for s in ks], name="Call OI", marker_color=RED), go.Bar(x=ks, y=[oc["pe"].get(s, (0, 0))[0] for s in ks], name="Put OI", marker_color=GREEN)])
    if cs["spot"]: fig.add_vline(x=cs["spot"], line_dash="dash", line_color=CYAN, annotation_text="spot")
    fig.update_layout(barmode="group", title="Open interest by strike"); st.plotly_chart(dark(fig, 400), width="stretch")
    fig = go.Figure([go.Bar(x=ks, y=[oc["ce"][s][1] for s in ks], name="Call OI change", marker_color=RED), go.Bar(x=ks, y=[oc["pe"].get(s, (0, 0))[1] for s in ks], name="Put OI change", marker_color=GREEN)]); fig.update_layout(barmode="group", title="Change in OI (fresh writing)"); st.plotly_chart(dark(fig, 300), width="stretch")
    sp = cs["spot"] or 0; msg = "Spot sits <b>between the walls</b> — range-trade bias until one breaks on volume." if cs["sup"] <= sp <= cs["res"] else "Spot is <b>outside the walls</b> — writers under pressure, trend move possible."
    logic(f"Heavy call OI above spot = resistance; heavy put OI below = support. {msg}")

elif page.startswith("🔌"):
    st.subheader("Data Sources — broker APIs, free sources and diagnostics")
    logic("<b>Priority in Auto mode:</b> 1) your broker API (if selected + credentials valid) → 2) Yahoo Finance (free) → 3) NSE bhavcopy archive (free, official daily data with delivery %; works when Yahoo is blocked) → 4) Demo. "
          "Every page shows a banner naming the source actually used. Pick <i>Broker API only</i> if you want a hard failure instead of a silent fallback.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Current configuration"); st.write(f"**Mode:** {mode_label}"); st.write(f"**Broker:** {BROKER or 'None'}")
        if BROKER: st.write("**Credentials entered:** " + ", ".join(f"{k} {'✔' if v else '✘'}" for k, v in creds.items()))
        if st.button("🔍 Test connection", type="primary"):
            with st.spinner("Testing…"):
                res = H.test_connection(CFG)
            st.dataframe(pd.DataFrame([{"Check": a, "Result": "✅ PASS" if b else "❌ FAIL", "Detail": c} for a, b, c in res]), hide_index=True, width="stretch")
    with c2:
        st.markdown("##### What each source provides")
        st.dataframe(pd.DataFrame([
            {"Source": "Dhan", "Daily": "✔", "Intraday 1–60m": "✔ (90d/call, years back)", "Option chain": "✔", "Auth": "client id + access token"},
            {"Source": "Angel One", "Daily": "✔", "Intraday 1–60m": "✔ (chunked)", "Option chain": "✘ (NSE fallback)", "Auth": "api key + client code + PIN + TOTP"},
            {"Source": "ICICI Breeze", "Daily": "✔", "Intraday 1–60m": "1/5/30m (15m built from 5m)", "Option chain": "✘ (NSE fallback)", "Auth": "api key + secret + daily session token"},
            {"Source": "Yahoo (free)", "Daily": "✔ (indices+stocks)", "Intraday 1–60m": "≤60 days", "Option chain": "✘", "Auth": "none"},
            {"Source": "NSE bhavcopy (free)", "Daily": "✔ + delivery %", "Intraday 1–60m": "✘", "Option chain": "live endpoint (cookie)", "Auth": "none"}]), hide_index=True, width="stretch")
    st.markdown("##### Setup")
    with st.expander("Dhan"): st.markdown("1. web.dhan.co → *My Profile → DhanHQ Trading APIs* → generate an **access token** (valid 24h; refresh daily or use the long-lived option).\n2. Paste **Client ID** and **Access token** in the sidebar.\n3. Data APIs may require the Data API plan on your account.")
    with st.expander("Angel One SmartAPI"): st.markdown("1. smartapi.angelone.in → create an app → **API key**.\n2. Enable TOTP and copy the **TOTP secret** shown beside the QR code.\n3. `pip install smartapi-python pyotp`; enter API key, client code, PIN, TOTP secret.")
    with st.expander("ICICI Direct Breeze"): st.markdown("1. api.icicidirect.com → create an app → **API key / secret**.\n2. Open the login URL `https://api.icicidirect.com/apiuser/login?api_key=<KEY>` and copy the **session token** from the redirect — it **expires daily**.\n3. `pip install breeze-connect`.")
    with st.expander("Keeping keys out of the UI (recommended for deployment)"): st.code('# .streamlit/secrets.toml  (never commit this file)\n[brokers."Dhan"]\nclient_id = "..."\naccess_token = "..."\n[brokers."Angel One"]\napi_key = "..."\nclient_code = "..."\npin = "..."\ntotp_secret = "..."\n[brokers."ICICI Breeze"]\napi_key = "..."\napi_secret = "..."\nsession_token = "..."', language="toml"); st.caption("Environment variables also work, e.g. DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN.")
    logic("<b>Verification status:</b> the three broker adapters follow each broker's published API and are unit-tested against mocked responses, but were <b>not</b> run against live accounts by the author. Use <i>Test connection</i> with your own keys; any series a broker can't serve is transparently taken from the next source and labelled.")

else:
    st.subheader("Logic Decoder — what each uploaded script does, and what changed")
    rows = [("app.py", "Streamlit v1: index snapshot, breadth, Heikin-Ashi + RSI<35 screener, MACD/EMA/RSI table, correlation matrix.", "All modules kept. HA 'green + RSI<limit' rebuilt as a colour-FLIP rebound and back-tested. Breadth gained history and a sector-rotation quadrant."),
            ("hilega_milega_pc.py", "Excel generator: HM state (RSI9/EMA3/WMA21/SMA50), D/W/M MTF, 5-EMA tags, weekly breakout, ATR/ADX, delivery %, swing/intraday scoring, 'win prob'.", "HM + weekly BO ported exactly; split/bonus back-adjustment ported for bhavcopy data. The original 'Win Prob %' was a linear map of a score — replaced with measured, shrunk win-rates. Weekly HM is shifted one week (no look-ahead)."),
            ("Market_Decoder.py", "FII/DII/Pro/Client OI, VIX-adaptive option-chain PCR/walls, US-futures global score, rule-based verdict, 3-day velocity.", "All kept; max-pain and OI-change added. The ±50k / 40–60% thresholds are heuristics and labelled so; they feed a bounded −10…+10 regime score."),
            ("nifty_screener_pro.py", "Fills an Excel planner from Yahoo/NSE; NR7 squeeze, volume burst, EMA20 dip-buy, bear-flag setups.", "Its 'score' was just ±50 (above/below SMA50). Setups kept as strategies and back-tested. Intraday part rebuilt in the Intraday Desk on real 15-minute data."),
            ("main.py", "Placeholder print statement.", "Replaced by app.py + engine / datahub / brokers / intraday / predict modules.")]
    for f, w, c in rows: st.markdown(f"##### 📄 {f}"); logic(f"<b>What it does:</b> {w}"); logic(f"<b>In this terminal:</b> {c}")
    st.markdown("#### Strategy library"); st.dataframe(pd.DataFrame([{"Key": s.key, "Strategy": s.name, "Source": s.source, "Rule": s.logic, "Why it might work": s.why} for s in E.STRATEGIES.values()]), hide_index=True, width="stretch")
    st.markdown("#### Honest limits")
    logic("• <b>No model or indicator can be 100% accurate.</b> Daily direction is near a coin flip in liquid markets; this app measures the edge instead of claiming it, and shows grade D when none exists.<br>"
          "• Back-tests are evidence, not promises; regimes change. Out-of-sample and baseline checks reduce, not remove, overfitting.<br>"
          "• Yahoo/bhavcopy are end-of-day, survivorship-biased (current constituents). Free intraday history is ≤60 days.<br>"
          "• Live NSE option-chain / participant-OI endpoints can block cloud IPs; the app labels any fallback.<br>"
          "• Broker adapters are untested against live accounts (mock-tested only).")
