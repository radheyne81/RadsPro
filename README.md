# NSE AI Smart Terminal v2
    pip install -r requirements.txt            # + requirements-brokers.txt for Angel One / Breeze
    streamlit run app.py
Files: app.py (UI) · engine.py (indicators, strategies, back-tests) · datahub.py (source chain) · brokers.py (Dhan / Angel One / ICICI Breeze)
· intraday.py (session factors, ORB/VWAP back-tests, next-session plan) · predict.py (walk-forward next-day model) · test_*.py
Data chain (Auto): Broker -> Yahoo -> NSE bhavcopy (free, official, has delivery %) -> Demo. Every page shows which source was used.
Sidebar -> Data source: choose mode + broker, paste credentials (memory only) or use .streamlit/secrets.toml.
Data Sources page -> "Test connection" verifies each feed with your keys.
Tests: python test_brokers.py ; python test_app.py   (mock-based; no credentials needed)
Deploy: GitHub -> Streamlit Community Cloud, main file app.py, put keys in Secrets.
