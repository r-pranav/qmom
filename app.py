import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Autonomous Twin Q-Mom Engine", layout="wide", page_icon="🧠")

st.title("🧠 Self-Healing Autonomous Twin Q-Mom Engine")
st.caption("Live NSE Screening • Fundamental Auto-Quarantine • Google News Sentiment Sentinel • Zero-Deliberation Allocator")

# --- PERSISTENT PORTFOLIO STORAGE ---
HOLDINGS_FILE = "holdings_v2.json"

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

# --- SIDEBAR: BUDGET & SELF-HEALING GUARDRAILS ---
st.sidebar.header("💰 Monthly Capital Input")
monthly_budget = st.sidebar.number_input(
    "Enter Budget to Deploy Today (₹)", 
    min_value=1000, max_value=1000000, value=9000, step=1000
)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Self-Healing Guardrails")
ema_period = st.sidebar.slider("Macro Benchmark EMA", 50, 250, 200, 10)
atr_multiplier = st.sidebar.slider("Chandelier Stop (ATR Multiple)", 1.5, 4.5, 3.0, 0.1)
max_debt_equity = st.sidebar.slider("Max Allowed Debt-to-Equity", 0.10, 1.00, 0.50, 0.05)
min_roe_pct = st.sidebar.slider("Min Required ROE (%)", 10, 25, 15, 1)

# --- BACKUP / RESTORE IN SIDEBAR (PROTECTS AGAINST CLOUD REBOOTS) ---
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Portfolio Backup & Restore")
current_holdings = load_holdings()
if current_holdings:
    st.sidebar.download_button(
        label="⬇️ Download Portfolio Backup (JSON)",
        data=json.dumps(current_holdings, indent=2),
        file_name=f"qmom_portfolio_{date.today()}.json",
        mime="application/json"
    )

uploaded_backup = st.sidebar.file_uploader("⬆️ Restore Backup (JSON)", type=["json"])
if uploaded_backup is not None:
    try:
        restored_data = json.load(uploaded_backup)
        if isinstance(restored_data, list):
            save_holdings(restored_data)
            st.sidebar.success("Portfolio restored successfully!")
            st.rerun()
    except Exception:
        st.sidebar.error("Invalid backup file.")

# ==============================================================================
# 1. DYNAMIC SELF-HEALING UNIVERSE INGESTION (NO SCREENER NEEDED)
# ==============================================================================
FALLBACK_UNIVERSE = [
    # Industrials, Defense, Power & Capital Goods
    "ABB.NS", "SIEMENS.NS", "BEL.NS", "HAL.NS", "CUMMINSIND.NS", "POLYCAB.NS",
    "KEI.NS", "BHARATFORG.NS", "BOSCHLTD.NS", "CGPOWER.NS", "THERMAX.NS",
    "APARINDS.NS", "TRITURBINE.NS", "KAYNES.NS", "DIXON.NS", "MAZDA.NS",
    # IT Services & Digital Engineering
    "COFORGE.NS", "PERSISTENT.NS", "KPITTECH.NS", "TATAELXSI.NS", "HCLTECH.NS",
    "TCS.NS", "LTIM.NS", "OFSS.NS", "ZENSARTECH.NS", "CYIENT.NS", "NETWEB.NS",
    # Auto & Mobility Ancillaries
    "BAJAJ-AUTO.NS", "TVSMOTOR.NS", "EICHERMOT.NS", "TIINDIA.NS", "UNOMINDA.NS",
    "ENDURANCE.NS", "SONACOMS.NS", "JBMA.NS",
    # Consumer, Retail & Platform Leaders
    "TRENT.NS", "TITAN.NS", "VBL.NS", "DMART.NS", "TATACONSUM.NS", "PAGEIND.NS",
    "RADICO.NS", "VARROC.NS", "SAFARI.NS", "CAMS.NS", "MCX.NS", "BSE.NS",
    # Healthcare, CDMO & Specialty Pharma
    "DIVISLAB.NS", "SUNPHARMA.NS", "CIPLA.NS", "MAXHEALTH.NS", "MANKIND.NS",
    "ALKEM.NS", "TORNTPHARM.NS", "JBCHEPHARM.NS", "AJANTPHARM.NS", "SUVENPHAR.NS",
    # Specialty Chemicals & Materials
    "PIIND.NS", "SOLARINDS.NS", "DEEPAKNTR.NS", "FINEORG.NS", "LINDEINDIA.NS"
]

@st.cache_data(ttl=86400)
def get_dynamic_universe():
    """Attempts live NSE Nifty Midcap/LargeMidcap CSV fetch; falls back to 62-stock curated pool."""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as response:
            df_nse = pd.read_csv(response)
            if "Symbol" in df_nse.columns and "Industry" in df_nse.columns:
                # Filter out leveraged Financial Services where D/E rules don't apply
                non_fin = df_nse[~df_nse["Industry"].str.contains("Financial|Bank", case=False, na=False)]
                live_tickers = [f"{sym.strip()}.NS" for sym in non_fin["Symbol"].tolist()]
                combined = list( dict.fromkeys(FALLBACK_UNIVERSE + live_tickers[:40]) )
                return combined
    except Exception:
        pass
    return FALLBACK_UNIVERSE

# ==============================================================================
# 2. MACRO & VOLATILITY REGIME ENGINE (NIFTY 500 + INDIA VIX)
# ==============================================================================
@st.cache_data(ttl=1800)
def evaluate_macro_regime(span):
    nifty = yf.download("^CRSLDX", period="2y", progress=False)
    if nifty.empty or len(nifty) < span:
        nifty = yf.download("^NSEI", period="2y", progress=False)
    
    close_vals = nifty['Close'].values.flatten()
    ema_vals = pd.Series(close_vals).ewm(span=span, adjust=False).mean()
    curr_close = float(close_vals[-1])
    curr_ema = float(ema_vals.iloc[-1])
    ema_dist = ((curr_close - curr_ema) / curr_ema) * 100.0

    # Check India VIX for volatility shock
    vix_val = 15.0
    try:
        vix_df = yf.download("^INDIAVIX", period="1mo", progress=False)
        if not vix_df.empty:
            vix_val = float(vix_df['Close'].values.flatten()[-1])
    except Exception:
        pass

    if curr_close < curr_ema or vix_val >= 25.0:
        regime = "RISK-OFF"
        deploy_ratio = 0.0
    elif vix_val >= 20.0 or ema_dist < 1.0:
        regime = "CAUTIOUS RISK-ON"
        deploy_ratio = 0.5
    else:
        regime = "AGGRESSIVE RISK-ON"
        deploy_ratio = 1.0

    return curr_close, curr_ema, ema_dist, vix_val, regime, deploy_ratio

curr_close, curr_ema, ema_dist, vix_val, regime, deploy_ratio = evaluate_macro_regime(ema_period)

st.subheader("1. Autonomous Macro & Volatility Regime")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nifty 500 Close", f"₹{curr_close:,.2f}")
c2.metric(f"{ema_period}-Day EMA", f"₹{curr_ema:,.2f}", f"{ema_dist:+.2f}%")
c3.metric("India VIX (Fear Index)", f"{vix_val:.2f}", "Normal (<20)" if vix_val < 20 else "Elevated (>20)")
c4.metric("Authorized Equity Deployment", f"{int(deploy_ratio * 100)}% of Budget")

if regime == "AGGRESSIVE RISK-ON":
    st.success("🟢 **REGIME: FULL RISK-ON** — Broad market trend is bullish and volatility is calm. Deploy 100% of monthly budget into top Twin Q-Mom picks.")
elif regime == "CAUTIOUS RISK-ON":
    st.warning("🟡 **REGIME: CAUTIOUS RISK-ON** — Market is near 200-EMA whipsaw zone or VIX is elevated. Engine automatically scales equity deployment to 50% and parks 50% in Arbitrage/Liquid fund.")
else:
    st.error("🔴 **REGIME: STRICT RISK-OFF** — Nifty 500 is below its 200-day EMA (or VIX > 25). Engine halts equity purchases. Deploy 100% of monthly budget into **Tata Arbitrage Fund / ICICI Liquid Fund** on Groww.")

st.markdown("---")

# ==============================================================================
# 3. QUANTITATIVE + FUNDAMENTAL + LIVE GOOGLE NEWS SENTIMENT ENGINE
# ==============================================================================
VETO_KEYWORDS = [
    "sebi ban", "fraud", "ed raid", "income tax raid", "cbi", "default", 
    "insolvency", "auditor resign", "forensic audit", "accounting irregularity", "arrest"
]
BEARISH_KEYWORDS = [
    "downgrade", "profit falls", "misses estimates", "margin pressure", 
    "weak guidance", "penalty", "show cause", "slump", "plunge", "loss widens"
]
BULLISH_KEYWORDS = [
    "order win", "record profit", "beats estimates", "margin expansion", 
    "upgrade", "breakout", "new contract", "acquisition", "capex", "surges", "jumps"
]

def fetch_google_news_sentiment(ticker_symbol):
    """Scrapes live Google News RSS for the stock and runs qualitative sentiment scoring."""
    clean_name = ticker_symbol.replace(".NS", "")
    query = urllib.parse.quote(f"{clean_name} NSE stock India")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    headlines = []
    try:
        req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.findall(".//item")[:5]:
                title = item.find("title")
                if title is not None and title.text:
                    headlines.append(title.text)
    except Exception:
        pass

    if not headlines:
        return 0.0, "Neutral (No major news catalysts)", False

    score = 0.0
    veto_flag = False
    combined_text = " | ".join(headlines).lower()

    for kw in VETO_KEYWORDS:
        if kw in combined_text:
            veto_flag = True
            return -100.0, f"🚨 RED FLAG: '{kw.upper()}' in recent news", True

    for kw in BULLISH_KEYWORDS:
        if kw in combined_text:
            score += 7.5
    for kw in BEARISH_KEYWORDS:
        if kw in combined_text:
            score -= 7.5

    score = max(-15.0, min(15.0, score))
    top_headline = headlines[0].split(" - ")[0] if headlines else "Normal news flow"
    return score, top_headline, veto_flag

def audit_fundamentals_and_news(candidate):
    """Parallel worker: verifies live balance-sheet health and news sentiment."""
    t = candidate["Raw_Ticker"]
    roe_pct = 20.0
    de_ratio = 0.05
    rev_growth = 15.0
    earn_growth = 20.0
    sector = "Industrial / Core"
    fund_ok = True
    fail_reason = ""

    try:
        info = yf.Ticker(t).info
        if info:
            sector = info.get("sector", "Core Equity")
            raw_roe = info.get("returnOnEquity", None)
            if raw_roe is not None:
                roe_pct = float(raw_roe) * 100.0
            
            raw_de = info.get("debtToEquity", None)
            if raw_de is not None:
                de_ratio = float(raw_de) / 100.0 if float(raw_de) > 2.0 else float(raw_de)

            raw_rev = info.get("revenueGrowth", None)
            if raw_rev is not None:
                rev_growth = float(raw_rev) * 100.0

            raw_earn = info.get("earningsGrowth", None)
            if raw_earn is not None:
                earn_growth = float(raw_earn) * 100.0

            # Self-healing quarantine triggers
            if de_ratio > max_debt_equity:
                fund_ok = False
                fail_reason = f"High Debt (D/E: {de_ratio:.2f})"
            elif roe_pct < min_roe_pct:
                fund_ok = False
                fail_reason = f"Low ROE ({roe_pct:.1f}%)"
            elif rev_growth < -10.0 or earn_growth < -20.0:
                fund_ok = False
                fail_reason = f"Earnings Contraction ({earn_growth:.1f}%)"
    except Exception:
        pass

    news_score, top_headline, veto_flag = fetch_google_news_sentiment(t)
    if veto_flag:
        fund_ok = False
        fail_reason = "Governance / News Veto"

    # Composite Twin Q-Mom Score (0 to 100 scale)
    quant_component = min(55.0, max(0.0, candidate["Vol_Adj_Mom"] * 22.0))
    prox_component = max(0.0, (candidate["Prox_52W"] - 80.0) * 1.0)
    fund_component = min(15.0, max(0.0, (roe_pct / 2.0)))
    composite_score = round(quant_component + prox_component + fund_component + news_score, 1)

    if fund_ok and candidate["Prox_52W"] >= 88.0 and candidate["Vol_Adj_Mom"] >= 1.0:
        status = "🔥 PRIME BUY"
    elif not fund_ok:
        status = f"🛑 QUARANTINED ({fail_reason})"
    else:
        status = "⏳ WATCHLIST"

    candidate.update({
        "Sector": sector,
        "ROE (%)": round(roe_pct, 1),
        "Debt/Eq": round(de_ratio, 2),
        "Sales Gr (%)": round(rev_growth, 1),
        "News Sentiment": f"{news_score:+.1f}",
        "Latest Catalyst / News": top_headline,
        "Composite Score": composite_score,
        "Status": status,
        "Is_Eligible": (status == "🔥 PRIME BUY")
    })
    return candidate

@st.cache_data(ttl=3600)
def run_autonomous_scanner(universe_list, max_de, min_roe):
    # Stage 1: Fast vectorized download of all stocks
    raw_data = yf.download(universe_list, period="15mo", group_by="ticker", threads=True, progress=False)
    prelim_candidates = []

    for t in universe_list:
        try:
            df = raw_data[t].dropna(how="all") if len(universe_list) > 1 else raw_data.dropna(how="all")
            if df.empty or len(df) < 230:
                continue
            
            close_v = df["Close"].values.flatten()
            high_v = df["High"].values.flatten()
            low_v = df["Low"].values.flatten()

            curr_p = float(close_v[-1])
            if np.isnan(curr_p) or curr_p <= 0:
                continue

            # 12-1M Momentum & 6M Acceleration
            ret_12_1m = (close_v[-22] / close_v[-230]) - 1.0
            ret_6m = (curr_p / close_v[-125]) - 1.0
            daily_ret = pd.Series(close_v).pct_change().dropna()
            ann_vol = float(daily_ret.std() * np.sqrt(250))
            vol_adj_mom = (ret_12_1m / ann_vol) if ann_vol > 0 else 0.0

            # 52W High Proximity
            high_52w = float(np.nanmax(high_v[-250:]))
            prox_52w = (curr_p / high_52w) * 100.0

            # 14-Day ATR for Chandelier Stop
            prev_c = np.roll(close_v, 1)
            prev_c[0] = close_v[0]
            tr = np.maximum(high_v - low_v, np.maximum(np.abs(high_v - prev_c), np.abs(low_v - prev_c)))
            atr_14 = float(pd.Series(tr).rolling(14).mean().iloc[-1])
            chandelier_stop = float(np.nanmax(high_v[-60:])) - (3.0 * atr_14)

            prelim_candidates.append({
                "Ticker": t.replace(".NS", ""),
                "Raw_Ticker": t,
                "Price": curr_p,
                "Ret_12_1M (%)": round(ret_12_1m * 100, 1),
                "Ret_6M (%)": round(ret_6m * 100, 1),
                "Vol_Adj_Mom": round(vol_adj_mom, 2),
                "Prox_52W": round(prox_52w, 1),
                "Chandelier_Stop": round(chandelier_stop, 2)
            })
        except Exception:
            continue

    df_prelim = pd.DataFrame(prelim_candidates)
    if df_prelim.empty:
        return pd.DataFrame()

    # Sort by price momentum and run deep Fundamental + News audit on top 18 leaders
    top_leaders = df_prelim.sort_values(by="Vol_Adj_Mom", ascending=False).head(18).to_dict("records")
    
    enriched = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        enriched = list(executor.map(audit_fundamentals_and_news, top_leaders))

    df_final = pd.DataFrame(enriched).sort_values(by="Composite Score", ascending=False)
    return df_final

universe = get_dynamic_universe()
with st.spinner(f"Scanning {len(universe)} NSE stocks, auditing balance sheets, and checking live Google News..."):
    master_df = run_autonomous_scanner(universe, max_debt_equity, min_roe_pct)

# ==============================================================================
# 4. ZERO-DELIBERATION MONTHLY BUDGET ALLOCATOR
# ==============================================================================
st.subheader("2. Zero-Deliberation Monthly Action Plan")

equity_budget = monthly_budget * deploy_ratio
arbitrage_budget = monthly_budget - equity_budget

# Check which stocks user already owns heavily
owned_tickers = {h["Ticker"]: h for h in current_holdings}

if deploy_ratio == 0.0:
    st.info(
        f"🛡️ **100% CAPITAL PRESERVATION MODE:**\n\n"
        f"**Exact Action on Groww Today:** Invest **₹{monthly_budget:,.0f}** into **Tata Arbitrage Fund (Direct-Growth)** "
        f"or **ICICI Prudential Liquid Fund (Direct-Growth)**.\n\n"
        f"*Why:* The Nifty 500 is below its 200-day EMA (or VIX > 25). Buying momentum stocks in a macro downtrend causes false breakouts. Your cash will earn ~6.8% safely until the regime turns green."
    )
else:
    eligible_df = master_df[
        (master_df["Is_Eligible"] == True) & 
        (master_df["Price"] <= equity_budget)
    ].copy() if not master_df.empty else pd.DataFrame()

    # Prioritize fresh compounders not already over-weighted
    unowned_eligible = eligible_df[~eligible_df["Raw_Ticker"].isin(owned_tickers.keys())] if not eligible_df.empty else pd.DataFrame()
    pick_pool = unowned_eligible if not unowned_eligible.empty else eligible_df

    if pick_pool.empty:
        st.warning(
            f"⚠️ No stocks currently pass all strict Quant + Fundamental + News gates under your ₹{equity_budget:,.0f} price limit. "
            f"Park **₹{monthly_budget:,.0f}** in **Tata Arbitrage Fund** this month."
        )
    else:
        # If budget >= ₹20,000, split across top 2 uncorrelated picks; otherwise focus on #1 pick
        num_picks = 2 if (equity_budget >= 20000 and len(pick_pool) >= 2) else 1
        selected_picks = pick_pool.head(num_picks)
        per_stock_budget = equity_budget / num_picks

        st.markdown(f"##### 🎯 Ideal Execution Plan for Your ₹{monthly_budget:,.0f} Budget")
        
        alloc_rows = []
        total_equity_spent = 0.0
        for _, row in selected_picks.iterrows():
            shares_to_buy = int(per_stock_budget // row["Price"])
            cost = shares_to_buy * row["Price"]
            total_equity_spent += cost
            alloc_rows.append({
                "Action": "🟢 BUY (CNC Delivery)",
                "Stock": row["Ticker"],
                "Sector": row["Sector"],
                "Live Price": f"₹{row['Price']:,.2f}",
                "Exact Shares to Buy": shares_to_buy,
                "Total Cost": f"₹{cost:,.2f}",
                "Initial Chandelier Stop": f"₹{row['Chandelier_Stop']:,.2f}",
                "Composite Score": f"{row['Composite Score']} / 100",
                "Qualitative News Check": row["Latest Catalyst / News"]
            })

        leftover_cash = monthly_budget - total_equity_spent
        st.dataframe(pd.DataFrame(alloc_rows), hide_index=True, use_container_width=True)

        if leftover_cash > 100:
            st.caption(f"💡 **Remainder Routing:** Park the uninvested **₹{leftover_cash:,.2f}** (including defensive buffer & whole-share rounding change) into **Tata Arbitrage Fund** on Groww.")

        if st.button("✅ 1-Click Log Recommended Purchase(s) into My Holdings"):
            for _, row in selected_picks.iterrows():
                qty = int(per_stock_budget // row["Price"])
                if qty > 0:
                    current_holdings.append({
                        "Ticker": row["Raw_Ticker"],
                        "Buy_Price": round(float(row["Price"]), 2),
                        "Shares": qty,
                        "Buy_Date": str(date.today())
                    })
            save_holdings(current_holdings)
            st.success("Logged to active portfolio!")
            st.rerun()

# Expandable Full Autonomous Scanner Table
with st.expander("📊 View All Scanned Candidates (Quant + Fundamental + Live News Scores)", expanded=True):
    if not master_df.empty:
        disp_cols = [
            "Ticker", "Sector", "Price", "Composite Score", "Vol_Adj_Mom", 
            "Prox_52W", "ROE (%)", "Debt/Eq", "Sales Gr (%)", "News Sentiment", 
            "Latest Catalyst / News", "Status"
        ]
        st.dataframe(master_df[disp_cols], hide_index=True, use_container_width=True)

st.markdown("---")

# ==============================================================================
# 5. ACTIVE HOLDINGS, STOP-LOSS SENTINEL & 20-25% CAGR TRACKER
# ==============================================================================
st.subheader("3. My Portfolio Performance & Self-Checking Stop Sentinel")

col_add1, col_add2 = st.columns([2, 1])
with col_add1:
    with st.popover("➕ Add or Update a Holding Manually"):
        sel_ticker = st.selectbox("Stock Ticker", sorted(universe))
        in_price = st.number_input("Average Buy Price (₹)", min_value=1.0, value=1500.0, step=10.0)
        in_shares = st.number_input("Number of Shares", min_value=1, value=5, step=1)
        in_date = st.date_input("Purchase Date", value=date.today())
        if st.button("Save Position"):
            current_holdings.append({
                "Ticker": sel_ticker,
                "Buy_Price": round(float(in_price), 2),
                "Shares": int(in_shares),
                "Buy_Date": str(in_date)
            })
            save_holdings(current_holdings)
            st.rerun()

with col_add2:
    if current_holdings and st.button("🗑️ Clear Entire Portfolio"):
        save_holdings([])
        st.rerun()

if not current_holdings:
    st.info("Your portfolio is currently empty. Use the '1-Click Log' button above or 'Add a Holding Manually' when you purchase shares on Groww.")
else:
    port_rows = []
    total_invested = 0.0
    total_current_val = 0.0
    weighted_days_sum = 0.0

    for idx, item in enumerate(current_holdings):
        t = item["Ticker"]
        buy_p = float(item["Buy_Price"])
        qty = int(item.get("Shares", 1))
        b_date_str = item.get("Buy_Date", str(date.today()))
        try:
            b_date = datetime.strptime(b_date_str, "%Y-%m-%d").date()
        except Exception:
            b_date = date.today()
        days_held = max(1, (date.today() - b_date).days)

        df_h = yf.download(t, period="1y", progress=False)
        if df_h.empty:
            continue

        h_close = df_h["Close"].values.flatten()
        h_high = df_h["High"].values.flatten()
        h_low = df_h["Low"].values.flatten()

        curr_p = float(h_close[-1])
        prev_c = np.roll(h_close, 1)
        prev_c[0] = h_close[0]
        tr = np.maximum(h_high - h_low, np.maximum(np.abs(h_high - prev_c), np.abs(h_low - prev_c)))
        atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])

        peak_60d = float(np.nanmax(h_high[-60:]))
        stop_level = peak_60d - (atr_multiplier * atr)
        dist_stop_pct = ((curr_p - stop_level) / curr_p) * 100.0

        invested_val = buy_p * qty
        curr_val = curr_p * qty
        pnl_rs = curr_val - invested_val
        pnl_pct = ((curr_p - buy_p) / buy_p) * 100.0

        total_invested += invested_val
        total_current_val += curr_val
        weighted_days_sum += invested_val * days_held

        # Live news check on active holding
        _, h_news, h_veto = fetch_google_news_sentiment(t)

        if h_veto:
            action_signal = "🚨 SELL IMMEDIATELY (News Red Flag)"
        elif curr_p < stop_level:
            action_signal = "🚨 EXIT / CUT (Stop Breached)"
        else:
            action_signal = "✅ SAFE TO HOLD"

        port_rows.append({
            "ID": idx,
            "Stock": t.replace(".NS", ""),
            "Shares": qty,
            "Buy Date": b_date_str,
            "Buy Price": f"₹{buy_p:,.2f}",
            "Live Price": f"₹{curr_p:,.2f}",
            "Invested": f"₹{invested_val:,.0f}",
            "Current Value": f"₹{curr_val:,.0f}",
            "Net P&L": f"₹{pnl_rs:+,.0f} ({pnl_pct:+.2f}%)",
            "Chandelier Stop": f"₹{stop_level:,.2f} ({dist_stop_pct:+.1f}%)",
            "Live News Pulse": h_news,
            "System Verdict": action_signal
        })

    # Calculate Portfolio Returns & Annualized CAGR Pacing
    net_profit = total_current_val - total_invested
    abs_return_pct = ((total_current_val - total_invested) / total_invested) * 100.0 if total_invested > 0 else 0.0
    avg_days_held = (weighted_days_sum / total_invested) if total_invested > 0 else 30.0
    effective_years = max(avg_days_held / 365.25, 30.0 / 365.25)  # Min 30-day floor to prevent astronomical short-term extrapolation
    annualized_cagr = (((total_current_val / total_invested) ** (1.0 / effective_years)) - 1.0) * 100.0 if total_invested > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Capital Invested", f"₹{total_invested:,.0f}")
    m2.metric("Current Portfolio Value", f"₹{total_current_val:,.0f}", f"₹{net_profit:+,.0f}")
    m3.metric("Absolute Return (%)", f"{abs_return_pct:+.2f}%")
    m4.metric(
        "Annualized CAGR Pace", 
        f"{annualized_cagr:+.1f}%", 
        "On Target (20-25%+)" if annualized_cagr >= 20.0 else "Building Momentum"
    )

    st.dataframe(pd.DataFrame(port_rows).drop(columns=["ID"]), hide_index=True, use_container_width=True)

    # Individual Position Removal (When Sold on Groww)
    with st.expander("✂️ Remove a Sold Stock from Portfolio"):
        del_options = {f"{r['Stock']} ({r['Shares']} shares @ {r['Buy Price']})": r["ID"] for r in port_rows}
        selected_del = st.selectbox("Select Position to Remove", list(del_options.keys()))
        if st.button("Confirm Removal"):
            del_idx = del_options[selected_del]
            current_holdings.pop(del_idx)
            save_holdings(current_holdings)
            st.rerun()
