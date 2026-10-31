import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

@st.cache_data(ttl=3600)  # Cache for 1 hour to save bandwidth
def get_benchmark_regime():
    # ^CRSLDX is the Yahoo ticker for Nifty 500
    nifty500 = yf.download("^CRSLDX", period="2y", progress=False)
    if nifty500.empty:
        # Fallback to Nifty 50 if Nifty 500 index ticker has connectivity issues
        nifty500 = yf.download("^NSEI", period="2y", progress=False)
    
    close = nifty500['Close'].squeeze()
    ema = close.ewm(span=ema_period, adjust=False).mean()
    current_close = float(close.iloc[-1])
    current_ema = float(ema.iloc[-1])
    diff_pct = ((current_close - current_ema) / current_ema) * 100
    
    return current_close, current_ema, diff_pct

current_close, current_ema, diff_pct = get_benchmark_regime()

col1, col2, col3 = st.columns(3)
col1.metric("Nifty Close", f"₹{current_close:,.2f}")
col2.metric(f"{ema_period}-Day EMA", f"₹{current_ema:,.2f}")
col3.metric("Distance from EMA", f"{diff_pct:+.2f}%")

if current_close >= current_ema:
    st.success("🟢 **REGIME: RISK-ON** — Market is in a confirmed uptrend. Safe to deploy fresh monthly SIP tranches into equity momentum.")
else:
    st.error("🔴 **REGIME: RISK-OFF** — Benchmark is trading below 200-day EMA. Pause fresh equity buying. Direct monthly SIP into Arbitrage/Liquid funds on Groww.")

st.markdown("---")

# --- 2. ACTIVE PORTFOLIO & CHANDELIER TRAILING STOPS ---
st.subheader("2. Active Holdings & Trailing Stops")

# Add or tweak your active holdings here
default_portfolio = [
    {"Ticker": "TCS.NS", "Entry_Price": 3800.0, "Weight": "12%"},
    {"Ticker": "HCLTECH.NS", "Entry_Price": 1650.0, "Weight": "12%"},
    {"Ticker": "BAJAJ-AUTO.NS", "Entry_Price": 8800.0, "Weight": "10%"},
    {"Ticker": "TITAN.NS", "Entry_Price": 3300.0, "Weight": "10%"},
    {"Ticker": "CUMMINSIND.NS", "Entry_Price": 3450.0, "Weight": "10%"}
]

def calculate_chandelier_stops(portfolio):
    data_list = []
    for item in portfolio:
        ticker = item["Ticker"]
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty:
            continue
        
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        close = df['Close'].squeeze()
        
        # 14-day True Range calculation
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_window).mean().iloc[-1]
        
        curr_price = float(close.iloc[-1])
        highest_high = float(high.iloc[-60:].max())  # 60-day rolling high
        stop_level = highest_high - (atr_multiplier * atr)
        status = "🚨 EXIT / CUT" if curr_price < stop_level else "✅ SAFE TO HOLD"
        
        data_list.append({
            "Ticker": ticker.replace(".NS", ""),
            "Current Price": round(curr_price, 2),
            "Highest High": round(highest_high, 2),
            "Chandelier Stop": round(stop_level, 2),
            "Distance to Stop": f"{((curr_price - stop_level) / curr_price)*100:+.2f}%",
            "Status": status
        })
    return pd.DataFrame(data_list)

portfolio_df = calculate_chandelier_stops(default_portfolio)
st.dataframe(portfolio_df, use_container_width=True)

st.markdown("---")

# --- 3. CANDIDATE SCREENER: 12-1M VOLATILITY-ADJUSTED MOMENTUM ---
st.subheader("3. Monthly Screener Pool (Vol-Adjusted Momentum)")

watchlist = [
    "DIXON.NS", "PERSISTENT.NS", "COFORGE.NS", "POLYCAB.NS", 
    "BEL.NS", "TRENT.NS", "HAL.NS", "ABB.NS", "SIEMENS.NS"
]

def screen_momentum(tickers):
    results = []
    for t in tickers:
        df = yf.download(t, period="15mo", progress=False)
        if len(df) < 250:
            continue
        close = df['Close'].squeeze()
        
        # 12-1 Month Momentum calculation (Return over 12 months excluding last month)
        ret_12m = (close.iloc[-22] / close.iloc[-250]) - 1
        daily_ret = close.pct_change()
        annual_vol = daily_ret.std() * np.sqrt(250)
        vol_adj_score = ret_12m / annual_vol if annual_vol > 0 else 0
        
        # Proximity to 52-Week High
        high_52w = float(df['High'].iloc[-250:].max())
        curr = float(close.iloc[-1])
        prox_52w = (curr / high_52w) * 100
        
        results.append({
            "Ticker": t.replace(".NS", ""),
            "Current Price": round(curr, 2),
            "12-1M Momentum Score": round(vol_adj_score, 2),
            "Proximity to 52W High": f"{prox_52w:.1f}%",
            "Action": "🔥 Prime Candidate" if prox_52w >= 90 and vol_adj_score > 1.2 else "Watch"
        })
    return pd.DataFrame(results).sort_values(by="12-1M Momentum Score", ascending=False)

screener_df = screen_momentum(watchlist)
st.dataframe(screener_df, use_container_width=True)
