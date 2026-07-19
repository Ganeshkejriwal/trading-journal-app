import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from google import genai

st.set_page_config(page_title="AI Trading Journal", layout="wide", page_icon="📈")

# ==========================================
# 1. MT5 & DATA LOGIC
# ==========================================
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

def fetch_mt5_data(login, password, server):
    # Fallback to mock data if MT5 isn't available (e.g., Mac/Linux) or credentials are empty
    if not MT5_AVAILABLE or not login:
        st.warning("MT5 library unavailable or credentials missing. Generating mock data for demonstration...")
        return generate_mock_trades()
    
    # Initialize MT5
    if not mt5.initialize():
        st.error("Failed to initialize MT5")
        return pd.DataFrame()
    
    # Login using Investor Password
    authorized = mt5.login(login, password=password, server=server)
    if not authorized:
        st.error(f"Failed to connect to MT5 server: {server}")
        return pd.DataFrame()
        
    # Get last 30 days of history
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=30)
    deals = mt5.history_deals_get(utc_from, utc_to)
    mt5.shutdown()
    
    if deals is None or len(deals) == 0:
        return pd.DataFrame()
        
    # Convert MT5 tuple data into a Pandas DataFrame
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Filter only closed trades (entry out) to calculate real profit
    df = df[df['entry'] == 1]
    return df

def generate_mock_trades():
    """Generates realistic fake trades to test the AI Coach."""
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=x, hours=np.random.randint(1, 12)) for x in range(30, 0, -1)]
    profits = np.random.normal(loc=10, scale=50, size=30)
    lots = np.random.choice([0.1, 0.5, 1.0], size=30)
    symbols = np.random.choice(['EURUSD', 'GBPUSD', 'XAUUSD'], size=30)
    
    df = pd.DataFrame({
        'time': dates,
        'symbol': symbols,
        'volume': lots,
        'profit': profits,
        'type': np.random.choice(['BUY', 'SELL'], size=30)
    })
    # Inject a "Revenge Trade" scenario for the AI to catch
    df.loc[28, 'profit'] = -150
    df.loc[29, 'time'] = df.loc[28, 'time'] + timedelta(minutes=2) # 2 mins later
    df.loc[29, 'volume'] = 2.0 # Doubled lot size
    df.loc[29, 'profit'] = -300
    
    return df.sort_values('time').reset_index(drop=True)

# ==========================================
# 2. AI COACH LOGIC
# ==========================================
def get_ai_feedback(df, api_key):
    if not api_key:
        return "Please enter your Gemini API Key in the sidebar to activate the AI Coach."
        
    client = genai.Client(api_key=api_key)
    
    # Calculate macro statistics to feed the AI context
    win_rate = len(df[df['profit'] > 0]) / len(df) * 100 if len(df) > 0 else 0
    net_profit = df['profit'].sum()
    
    summary_stats = f"""
    Total Trades: {len(df)}
    Win Rate: {win_rate:.1f}%
    Net Profit: ${net_profit:.2f}
    """
    
    # Pass the last 15 trades for micro-behavior analysis
    recent_trades = df.tail(15).to_csv(index=False)
    
    prompt = f"""
    You are an expert, strict, and highly analytical trading coach. 
    Analyze this trader's recent performance and identify specific behavioral flaws like 'revenge trading' 
    (taking large lots immediately after a loss), time-of-day weaknesses, or over-leveraging.

    Performance Summary:
    {summary_stats}

    Recent Trades Data (CSV):
    {recent_trades}

    Give actionable, blunt advice. Highlight specific trades if they made a mistake (like doubling down on a loss). 
    Do NOT give generic advice. Base it entirely on the data provided.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error contacting AI: {str(e)}"

# ==========================================
# 3. USER INTERFACE (DASHBOARD)
# ==========================================
def main():
    st.title("🧠 AI-Powered Trading Journal")
    
    # Sidebar - Settings & Sync
    with st.sidebar:
        st.header("1. MT5 Connection")
        mt5_login = st.text_input("MT5 Login ID")
        mt5_pass = st.text_input("Investor Password", type="password")
        mt5_server = st.text_input("Broker Server", "MetaQuotes-Demo")
        
        st.header("2. AI Coach Setup")
        gemini_key = st.text_input("Gemini API Key", type="password", help="Get this from Google AI Studio")
        
        if st.button("Sync Trades & Analyze", type="primary"):
            with st.spinner("Fetching trades from MT5..."):
                login_id = int(mt5_login) if mt5_login else 0
                st.session_state['trades'] = fetch_mt5_data(login_id, mt5_pass, mt5_server)
            
            if not st.session_state['trades'].empty:
                with st.spinner("AI is analyzing your behavior..."):
                    st.session_state['ai_feedback'] = get_ai_feedback(st.session_state['trades'], gemini_key)

    # Main Content Area
    if 'trades' not in st.session_state or st.session_state['trades'].empty:
        st.info("👈 Enter your details in the sidebar and click 'Sync Trades' to build your dashboard.")
        return
        
    df = st.session_state['trades']
    
    # Top Level Metrics
    wins = df[df['profit'] > 0]
    losses = df[df['profit'] <= 0]
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
    net_profit = df['profit'].sum()
    profit_factor = abs(wins['profit'].sum() / losses['profit'].sum()) if losses['profit'].sum() != 0 else float('inf')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Profit", f"${net_profit:.2f}", delta=f"{len(df)} Trades")
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Profit Factor", f"{profit_factor:.2f}")
    col4.metric("Avg Risk (Lot Size)", f"{df['volume'].mean():.2f}")

    # Application Tabs
    tab1, tab2, tab3 = st.tabs(["🤖 AI Coach", "📊 Equity Curve", "📝 Raw Data"])
    
    with tab1:
        st.subheader("Your Personal AI Trading Coach")
        if 'ai_feedback' in st.session_state:
            st.markdown(st.session_state['ai_feedback'])
        else:
            st.warning("AI analysis failed or API key missing.")
            
    with tab2:
        st.subheader("Account Growth")
        df['cumulative_profit'] = df['profit'].cumsum()
        fig = px.line(df, x='time', y='cumulative_profit', title="Equity Curve (Last 30 Days)", markers=True)
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.subheader("Recent Deals")
        st.dataframe(df.sort_values('time', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
