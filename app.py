import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Auto Q-Mom Command Center", layout="wide", page_icon="⚡")

st.title("⚡ Autonomous Asymmetric Twin-Momentum Command Center")
st.caption("Fully automated screening, stock selection, and portfolio tracking — zero manual code editing.")

# --- PERSISTENT HOLDINGS STORAGE (NO CODE EDITING) ---
HOLDINGS_FILE = "holdings.json"

def load_holdings():
    if os.path.exists(HOLDINGS_FILE):
        try:
            with open(HOLDINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_holdings(holdings):
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(holdings, f, indent=2)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Strategy Parameters")
ema_period = st.sidebar.slider("Benchmark EMA Filter", min_value=50, max_value=250, value=200, step=10)
atr_multiplier = st.sidebar.slider("Chandelier ATR Multiplier", min_value=1.5, max_value=4.0, value=3.0, step=0.1)
atr_window = st.sidebar.slider("ATR Lookback (Days)", min_value=10, max_value=30, value=14, step=1)
budget_input = st.sidebar.number_input("Monthly Capital to Deploy (₹)", min_value=1000, max_value=500000, value=9000, step=1000)

# --- EMBEDDED PRE-VETTED QUALITY MASTER UNIVERSE ---
# Pre-filtered for: Market Cap > ₹10,000 Cr, D/E < 0.20, ROE > 18%, pristine capital allocation
QUALITY_UNIVERSE = [
    # Industrials, Defense & Capital Goods
    "ABB.NS", "SIEMENS.NS", "BEL.NS", "HAL.NS", "CUMMINSIND.NS", 
    "POLYCAB.NS", "KEI.NS", "BHARATFORG.NS", "BOSCHLTD.NS",
    # IT Services & Digital ER&D
    "COFORGE.NS", "PERSISTENT.NS", "KPITTECH.NS", "TATAELXSI.NS", 
    "DIXON.NS", "HCLTECH.NS", "TCS.NS", "LTIM.NS",
    # Auto & Ancillaries
    "BAJAJ-AUTO.NS", "TVSMOTOR.NS", "MOTHERSON.NS", "TIINDIA.NS",
    # Consumer Staples, Discretionary & Retail
    "TRENT.NS", "TITAN.NS", "VBL.NS", "DMART.NS", "TATACONSUM.NS", "PAGEIND.NS",
    # Healthcare & Specialty Pharma
    "DIVISLAB.NS", "SUNPHARMA.NS", "CIPLA.NS", "MAXHEALTH.NS", "MANKIND.NS"
]

# ==============================================================================
# SECTION 1: MACRO REGIME CHECK (NIFTY 500)
# ==============================================================================
st.subheader("1. Macro Regime Filter (Nifty 500)")

@st.cache_data(ttl=3600)
def get_benchmark_regime(period):
    nifty = yf.download("^CRSLDX", period="2y", progress=False)
    if nifty.empty or len(nifty) < period:
        nifty = yf.download("^NSEI", period="2y", progress=False)
    
    close_vals = nifty['Close'].values.flatten()
    close_series = pd.Series(close_vals)
    ema_series = close_series.ewm(span=period, adjust=False).mean()
    
    current_close = float(close_vals[-1])
    current_ema = float(ema_series.iloc[-1])
    diff_pct = ((current_close - current_ema) / current_ema) * 100
    
    return current_close, current_ema, diff_pct

current_close, current_ema, diff_pct = get_benchmark_regime(ema_period)
is_risk_on = current_close >= current_ema

col1, col2, col3 = st.columns(3)
col1.metric("Nifty 500 Close", f"₹{current_close:,.2f}")
col2.metric(f"{ema_period}-Day EMA", f"₹{current_ema:,.2f}")
col3.metric("Distance from 200-EMA", f"{diff_pct:+.2f}%")

if is_risk_on:
    st.success("🟢 **REGIME: RISK-ON** — Market in structural uptrend. Safe to deploy fresh equity tranches.")
else:
    st.error("🔴 **REGIME: RISK-OFF** — Benchmark is below 200-day EMA. Halt equity buying. Park budget in Arbitrage/Liquid fund on Groww.")

st.markdown("---")

# ==============================================================================
# SECTION 2: AUTOMATIC QUANTITATIVE SCREENER & ORDER GENERATOR
# ==============================================================================
st.subheader("2. Automated Twin-Momentum Screener & Model Picks")

@st.cache_data(ttl=3600)
def evaluate_universe(universe):
    screen_data = []
    for ticker in universe:
        df = yf.download(ticker, period="15mo", progress=False)
        if df.empty or len(df) < 250:
            continue
        
        close_vals = df['Close'].values.flatten()
        high_vals = df['High'].values.flatten()
        
        # 12-1M Volatility-Adjusted Momentum
        ret_12m = (close_vals[-22] / close_vals[-250]) - 1.0
        daily_ret = pd.Series(close_vals).pct_change().dropna()
        annual_vol = float(daily_ret.std() * np.sqrt(250))
        vol_score = (ret_12m / annual_vol) if annual_vol > 0 else 0.0
        
        # 52-Week High Proximity
        high_52w = float(np.nanmax(high_vals[-250:]))
        curr_price = float(close_vals[-1])
        prox_52w = (curr_price / high_52w) * 100.0
        
        is_prime = (prox_52w >= 90.0) and (vol_score >= 1.0)
        
        screen_data.append({
            "Ticker": ticker.replace(".NS", ""),
            "Raw_Ticker": ticker,
            "Price": curr_price,
            "Momentum_Score": vol_score,
            "Proximity_52W": prox_52w,
            "Status": "🔥 Prime Candidate" if is_prime else "Watch"
        })
        
    df_all = pd.DataFrame(screen_data)
    if not df_all.empty:
        df_all = df_all.sort_values(by="Momentum_Score", ascending=False)
    return df_all

with st.spinner("Analyzing universe price velocity, volatility, and 52-week highs..."):
    screen_df = evaluate_universe(QUALITY_UNIVERSE)

prime_candidates = screen_df[screen_df["Status"] == "🔥 Prime Candidate"].head(4)

# Top Model Picks & Groww Order Sheet
st.markdown("##### 🎯 Top Model Selections & Exact Groww Order Sheet")
if not is_risk_on:
    st.info("ℹ️ **Market is Risk-Off:** Do not execute orders. Keep capital parked in Arbitrage/Liquid funds.")
elif prime_candidates.empty:
    st.warning("⚠️ No stocks currently meet the strict 12-1M Momentum (>1.0) and 52W-High (>=90%) criteria. Hold in cash/arbitrage.")
else:
    allocation_per_stock = budget_input / len(prime_candidates)
    order_sheet = []
    
    for _, row in prime_candidates.iterrows():
        qty = int(allocation_per_stock // row["Price"])
        actual_val = qty * row["Price"]
        order_sheet.append({
            "Stock": row["Ticker"],
            "Current Price": f"₹{row['Price']:,.2f}",
            "Allocated Budget": f"₹{allocation_per_stock:,.0f}",
            "Shares to Buy (CNC)": qty,
            "Total Trade Value": f"₹{actual_val:,.2f}",
            "Score": f"{row['Momentum_Score']:.2f}",
            "52W High Prox": f"{row['Proximity_52W']:.1f}%"
        })
    st.dataframe(pd.DataFrame(order_sheet), hide_index=True, use_container_width=True)

# Full Universe Table in an expander
with st.expander("🔍 View Complete 32-Stock Master Universe Screen"):
    display_df = screen_df.copy()
    display_df["Price"] = display_df["Price"].map(lambda p: f"₹{p:,.2f}")
    display_df["Momentum_Score"] = display_df["Momentum_Score"].map(lambda s: f"{s:.2f}")
    display_df["Proximity_52W"] = display_df["Proximity_52W"].map(lambda p: f"{p:.1f}%")
    st.dataframe(display_df[["Ticker", "Price", "Momentum_Score", "Proximity_52W", "Status"]], hide_index=True, use_container_width=True)

st.markdown("---")

# ==============================================================================
# SECTION 3: IN-BROWSER INTERACTIVE PORTFOLIO & TRAILING STOPS
# ==============================================================================
st.subheader("3. My Active Holdings & Trailing Stop Guardrails")
st.caption("Manage holdings directly in your browser. No code or GitHub editing required.")

current_holdings = load_holdings()

# Action Bar: Sync with Model or Manually Add
col_act1, col_act2 = st.columns([1, 1])

with col_act1:
    if st.button("📥 Adopt Top Model Picks as Active Holdings"):
        if not prime_candidates.empty:
            new_holdings = []
            for _, r in prime_candidates.iterrows():
                new_holdings.append({
                    "Ticker": r["Raw_Ticker"],
                    "Buy_Price": round(r["Price"], 2),
                    "Shares": int((budget_input / len(prime_candidates)) // r["Price"])
                })
            save_holdings(new_holdings)
            st.success("Holdings updated! Refreshing...")
            st.rerun()

with col_act2:
    with st.popover("➕ Add Custom Holding"):
        add_ticker = st.selectbox("Select Stock", QUALITY_UNIVERSE)
        add_price = st.number_input("Entry Price (₹)", min_value=1.0, value=1000.0, step=10.0)
        add_qty = st.number_input("Shares Owned", min_value=1, value=5, step=1)
        if st.button("Save Holding"):
            current_holdings.append({"Ticker": add_ticker, "Buy_Price": add_price, "Shares": add_qty})
            save_holdings(current_holdings)
            st.success("Stock added to portfolio!")
            st.rerun()

# Calculate Chandelier Trailing Stops for Active Holdings
if not current_holdings:
    st.info("No active holdings saved. Once you buy stocks on Groww, click 'Adopt Top Model Picks' or add them above.")
else:
    holdings_eval = []
    for item in current_holdings:
        ticker = item["Ticker"]
        df_h = yf.download(ticker, period="1y", progress=False)
        if df_h.empty or len(df_h) < atr_window + 1:
            continue
        
        h_high = df_h['High'].values.flatten()
        h_low = df_h['Low'].values.flatten()
        h_close = df_h['Close'].values.flatten()
        
        # 14-day ATR
        prev_c = np.roll(h_close, 1)
        prev_c[0] = h_close[0]
        tr = np.maximum(h_high - h_low, np.maximum(np.abs(h_high - prev_c), np.abs(h_low - prev_c)))
        atr = float(pd.Series(tr).rolling(window=atr_window).mean().iloc[-1])
        
        curr_p = float(h_close[-1])
        peak_high = float(np.nanmax(h_high[-60:]))
        stop_val = peak_high - (atr_multiplier * atr)
        pnl_pct = ((curr_p - item["Buy_Price"]) / item["Buy_Price"]) * 100
        stop_dist = ((curr_p - stop_val) / curr_p) * 100
        
        status = "🚨 EXIT / CUT" if curr_p < stop_val else "✅ SAFE TO HOLD"
        
        holdings_eval.append({
            "Ticker": ticker.replace(".NS", ""),
            "Shares": item.get("Shares", 1),
            "Buy Price": f"₹{item['Buy_Price']:,.2f}",
            "Current Price": f"₹{curr_p:,.2f}",
            "P&L (%)": f"{pnl_pct:+.2f}%",
            "Chandelier Stop": f"₹{stop_val:,.2f}",
            "Dist to Stop": f"{stop_dist:+.2f}%",
            "Status": status
        })
        
    st.dataframe(pd.DataFrame(holdings_eval), hide_index=True, use_container_width=True)
    
    if st.button("🗑️ Clear All Active Holdings"):
        save_holdings([])
        st.rerun()
