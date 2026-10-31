import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Q-Mom Asymmetric Tracker", layout="wide", page_icon="📈")

st.title("🛡️ Asymmetric Twin-Momentum Command Center")
st.caption("Free, open-source factor tracker for Indian Equities (NSE)")

# --- SIDEBAR: TWEAKABLE STRATEGY PARAMETERS ---
st.sidebar.header("⚙️ Strategy Parameters")
ema_period = st.sidebar.slider("Benchmark EMA Trend Filter", min_value=50, max_value=250, value=200, step=10)
atr_multiplier = st.sidebar.slider("Chandelier ATR Multiplier", min_value=1.5, max_value=4.0, value=3.0, step=0.1)
atr_window = st.sidebar.slider("ATR Calculation Window (Days)", min_value=10, max_value=30, value=14, step=1)

# --- 1. MACRO REGIME CHECK (NIFTY 500) ---
st.subheader("1. Macro Regime Filter (Nifty 500)")

@st.cache_data(ttl=3600)
def get_benchmark_regime():
    # Primary: Nifty 500 (^CRSLDX), Fallback: Nifty 50 (^NSEI)
    nifty = yf.download("^CRSLDX", period="2y", progress=False)
    if nifty.empty or len(nifty) < ema_period:
        nifty = yf.download("^NSEI", period="2y", progress=False)
    
    close_vals = nifty['Close'].values.flatten()
    close_series = pd.Series(close_vals)
    ema_series = close_series.ewm(span=ema_period, adjust=False).mean()
    
    current_close = float(close_vals[-1])
    current_ema = float(ema_series.iloc[-1])
    diff_pct = ((current_close - current_ema) / current_ema) * 100
    
    return current_close, current_ema, diff_pct

current_close, current_ema, diff_pct = get_benchmark_regime()

col1, col2, col3 = st.columns(3)
col1.metric("Nifty Close", f"₹{current_close:,.2f}")
col2.metric(f"{ema_period}-Day EMA", f"₹{current_ema:,.2f}")
col3.metric("Distance from EMA", f"{diff_pct:+.2f}%")

if current_close >= current_ema:
    st.success("🟢 **REGIME: RISK-ON** — Benchmark is above 200-EMA. Safe to deploy fresh monthly equity SIP tranches.")
else:
    st.error("🔴 **REGIME: RISK-OFF** — Benchmark is trading below 200-day EMA. Pause fresh equity buying. Direct monthly SIP into Arbitrage/Liquid funds on Groww.")

st.markdown("---")

# --- 2. ACTIVE HOLDINGS & TRAILING STOPS ---
st.subheader("2. Active Holdings & Trailing Stops")

default_portfolio = [
    {"Ticker": "TCS.NS", "Entry_Price": 3800.0, "Weight": "12%"},
    {"Ticker": "HCLTECH.NS", "Entry_Price": 1650.0, "Weight": "12%"},
    {"Ticker": "BAJAJ-AUTO.NS", "Entry_Price": 8800.0, "Weight": "10%"},
    {"Ticker": "TITAN.NS", "Entry_Price": 3300.0, "Weight": "10%"},
    {"Ticker": "CUMMINSIND.NS", "Entry_Price": 3450.0, "Weight": "10%"}
]

@st.cache_data(ttl=1800)
def calculate_chandelier_stops(portfolio):
    data_list = []
    for item in portfolio:
        ticker = item["Ticker"]
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty or len(df) < atr_window + 1:
            continue
        
        high_vals = df['High'].values.flatten()
        low_vals = df['Low'].values.flatten()
        close_vals = df['Close'].values.flatten()
        
        # 14-day True Range calculation
        prev_close = np.roll(close_vals, 1)
        prev_close[0] = close_vals[0]
        
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - prev_close)
        tr3 = np.abs(low_vals - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = float(pd.Series(tr).rolling(window=atr_window).mean().iloc[-1])
        curr_price = float(close_vals[-1])
        highest_high = float(np.nanmax(high_vals[-60:]))  # 60-day peak
        stop_level = highest_high - (atr_multiplier * atr)
        dist_to_stop = ((curr_price - stop_level) / curr_price) * 100
        
        status = "🚨 EXIT / CUT" if curr_price < stop_level else "✅ SAFE TO HOLD"
        
        data_list.append({
            "Ticker": ticker.replace(".NS", ""),
            "Current Price": f"₹{curr_price:,.2f}",
            "Highest High": f"₹{highest_high:,.2f}",
            "Chandelier Stop": f"₹{stop_level:,.2f}",
            "Distance to Stop": f"{dist_to_stop:+.2f}%",
            "Status": status
        })
    return pd.DataFrame(data_list)

portfolio_df = calculate_chandelier_stops(default_portfolio)
st.dataframe(portfolio_df, hide_index=True, use_container_width=True)

st.markdown("---")

# --- 3. CANDIDATE SCREENER: 12-1M VOLATILITY-ADJUSTED MOMENTUM ---
st.subheader("3. Monthly Screener Pool (Vol-Adjusted Momentum)")

watchlist = [
    "DIXON.NS", "PERSISTENT.NS", "COFORGE.NS", "POLYCAB.NS", 
    "BEL.NS", "TRENT.NS", "HAL.NS", "ABB.NS", "SIEMENS.NS"
]

@st.cache_data(ttl=3600)
def screen_momentum(tickers):
    results = []
    for t in tickers:
        df = yf.download(t, period="15mo", progress=False)
        if df.empty or len(df) < 250:
            continue
        
        close_vals = df['Close'].values.flatten()
        high_vals = df['High'].values.flatten()
        
        # 12-1 Month Momentum (Return over 12M excluding recent 22 trading days)
        ret_12m = (close_vals[-22] / close_vals[-250]) - 1.0
        daily_ret = pd.Series(close_vals).pct_change().dropna()
        annual_vol = float(daily_ret.std() * np.sqrt(250))
        vol_adj_score = (ret_12m / annual_vol) if annual_vol > 0 else 0.0
        
        # Proximity to 52-Week High (Using flat numpy max to avoid TypeError)
        high_52w = float(np.nanmax(high_vals[-250:]))
        curr = float(close_vals[-1])
        prox_52w = (curr / high_52w) * 100.0
        
        results.append({
            "Ticker": t.replace(".NS", ""),
            "Current Price": f"₹{curr:,.2f}",
            "12-1M Momentum Score": round(vol_adj_score, 2),
            "Proximity to 52W High": f"{prox_52w:.1f}%",
            "Action": "🔥 Prime Candidate" if prox_52w >= 90.0 and vol_adj_score > 1.2 else "Watch"
        })
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by="12-1M Momentum Score", ascending=False)
    return df_res

screener_df = screen_momentum(watchlist)
st.dataframe(screener_df, hide_index=True, use_container_width=True)
