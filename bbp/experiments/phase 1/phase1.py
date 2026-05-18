import sys
import os
import time
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from env.target_system import HotellingMarket

# Page config
st.set_page_config(
    page_title="Market Simulation",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3F9AAE, #F96E5B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .chart-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #444;
        margin-bottom: 0.5rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .firm-a-color { color: #3F9AAE; font-weight: 600; }
    .firm-b-color { color: #F96E5B; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

def moving_average(series, window_size=10):
    return series.rolling(window=window_size).mean()

def create_pricing_inputs(column, firm_name, regime, color_class, default_uniform=2.0, default_new=1.5, default_old=2.5):
    """Create pricing input widgets based on regime selection."""
    if regime == 0:  # Uniform Pricing
        return {
            'uniform_price': column.number_input(
                label=f"💰 {firm_name} Uniform Price",
                min_value=0.0,
                value=default_uniform,
                step=0.1,
                help=f"Set the uniform price for {firm_name}"
            ),
            'price_new': None,
            'price_old': None
        }
    else:  # Behavior-Based Pricing
        return {
            'uniform_price': None,
            'price_new': column.number_input(
                label=f"🆕 {firm_name} New Customer Price",
                min_value=0.0,
                value=default_new,
                step=0.1,
                help=f"Set the price for new customers for {firm_name}"
            ),
            'price_old': column.number_input(
                label=f"🔄 {firm_name} Returning Customer Price",
                min_value=0.0,
                value=default_old,
                step=0.1,
                help=f"Set the price for returning customers for {firm_name}"
            )
        }

# Configuration
REGIME_OPTIONS = {
    0: "Uniform Pricing",
    1: "Behavior-Based Pricing"
}

# Colors
FIRM_A_COLOR = '#3F9AAE'
FIRM_B_COLOR = '#F96E5B'

# Main UI
st.markdown('<h1 class="main-header">Live Market Simulation</h1>', unsafe_allow_html=True)
st.markdown("---")

# Settings in sidebar for cleaner look
with st.sidebar:
    st.markdown("## ⚙️ Experiment Settings")
    
    st.markdown("### 🏢 Firm A")
    regime_a = st.segmented_control(
        label="Pricing Strategy",
        options=REGIME_OPTIONS.keys(),
        default=0,
        format_func=lambda x: REGIME_OPTIONS[x],
        selection_mode="single",
        key="regime_a"
    )
    prices_a = create_pricing_inputs(st, "Firm A", regime_a, "firm-a-color")
    
    st.markdown("---")
    
    st.markdown("### 🏭 Firm B")
    regime_b = st.segmented_control(
        label="Pricing Strategy",
        options=REGIME_OPTIONS.keys(),
        default=0,
        format_func=lambda x: REGIME_OPTIONS[x],
        selection_mode="single",
        key="regime_b"
    )
    prices_b = create_pricing_inputs(st, "Firm B", regime_b, "firm-b-color")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Market Parameters")
    n_consumers = st.number_input(
        label="👥 Number of Consumers",
        min_value=1,
        value=1000,
        step=10,
        help="Total consumers in the market"
    )
    
    exclusivity = st.slider(
        label="✨ Exclusivity Seekness", 
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Mean of the exclusivity distribution"
    )
    
    strategicness = st.slider(
        label="🧠 Strategicness", 
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Mean of the strategicness distribution"
    )

# Initialize market
market = HotellingMarket(n_consumers, 42, exclusivity, strategicness)
market.set_regimes(regime_a, regime_b)

# Metrics row
st.markdown("### 📈 Real-Time Metrics")
metric_cols = st.columns(4)
metric_a_share = metric_cols[0].empty()
metric_b_share = metric_cols[1].empty()
metric_a_profit = metric_cols[2].empty()
metric_b_profit = metric_cols[3].empty()

# Chart layout
st.markdown("---")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 📊 Market Share")
    market_share_data = pd.DataFrame({"Firm A": [], "Firm B": []})
    chart_share = st.line_chart(market_share_data, color=[FIRM_A_COLOR, FIRM_B_COLOR], height=250)
    
    st.markdown("#### 💵 Period Profits")
    profit_data = pd.DataFrame({"Firm A": [], "Firm B": []})
    chart_profit = st.line_chart(profit_data, color=[FIRM_A_COLOR, FIRM_B_COLOR], height=250)

with chart_col2:
    st.markdown("#### 📉 Market Share (Moving Avg)")
    moving_avg_data = pd.DataFrame({"Firm A": [], "Firm B": []})
    chart_avg = st.line_chart(moving_avg_data, color=[FIRM_A_COLOR, FIRM_B_COLOR], height=250)
    
    st.markdown("#### 💰 Cumulative Profits")
    cumulative_data = pd.DataFrame({"Firm A": [], "Firm B": []})
    chart_cumulative = st.line_chart(cumulative_data, color=[FIRM_A_COLOR, FIRM_B_COLOR], height=250)

# Progress bar
progress_bar = st.progress(0, text="Simulation starting...")

# Run simulation
for step in range(100):
    market.step(prices_a, prices_b)
    
    # Calculate values
    firm_a_avg = moving_average(pd.Series(market.firms[0].market_share_history))
    firm_b_avg = moving_average(pd.Series(market.firms[1].market_share_history))
    firm_a_state,firm_b_state = market.get_firm_state(0), market.get_firm_state(1)
    firm_a_share , firm_b_share =firm_a_state["market_share"], firm_b_state["market_share"]
    firm_a_popularity_change, firm_b_popularity_change = firm_a_state["popularity_change"], firm_b_state["popularity_change"]
    firm_a_retention_rate, firm_b_retention_rate = firm_a_state["retention_rate"], firm_b_state["retention_rate"]
    firm_a_relative_popularity, firm_b_relative_popularity = firm_a_state["relative_popularity"], firm_b_state["relative_popularity"]
    firm_a_new_old_ratio, firm_b_new_old_ratio = firm_a_state["new_old_ratio"], firm_b_state["new_old_ratio"]
    firm_a_last_demand, firm_b_last_demand = firm_a_state["last_demand"], firm_b_state["last_demand"]
    firm_a_period_profit, firm_b_period_profit = firm_a_state["period_profit"], firm_b_state["period_profit"]
    firm_a_cumulative_profit, firm_b_cumulative_profit = firm_a_state["cumulative_profit"], firm_b_state["cumulative_profit"]
    firm_a_demand_new, firm_b_demand_new = firm_a_state["demand_new"], firm_b_state["demand_new"]
    firm_a_demand_old, firm_b_demand_old = firm_a_state["demand_old"], firm_b_state["demand_old"]

    
    # Update metrics
    metric_a_share.metric("🏢 Firm A Share", f"{firm_a_share:.1%}", 
                          delta=f"{firm_a_share - 0.5:+.1%}" if step > 0 else None)
    metric_b_share.metric("🏭 Firm B Share", f"{firm_b_share:.1%}",
                          delta=f"{firm_b_share - 0.5:+.1%}" if step > 0 else None)
    metric_a_profit.metric("💵 Firm A Profit", f"${firm_a_period_profit:,.0f}")
    metric_b_profit.metric("💰 Firm B Profit", f"${firm_b_period_profit:,.0f}")
    
    # Update charts
    chart_share.add_rows(pd.DataFrame({"Firm A": [firm_a_share], "Firm B": [firm_b_share]}))
    chart_avg.add_rows(pd.DataFrame({"Firm A": [firm_a_avg.iloc[-1]], "Firm B": [firm_b_avg.iloc[-1]]}))
    chart_profit.add_rows(pd.DataFrame({"Firm A": [firm_a_period_profit], "Firm B": [firm_b_period_profit]}))
    chart_cumulative.add_rows(pd.DataFrame({"Firm A": [firm_a_cumulative_profit], "Firm B": [firm_b_cumulative_profit]}))
    
    # Update progress
    progress_bar.progress((step + 1) / 100, text=f"⏳ Simulation Progress: Step {step + 1}/100")
    
    time.sleep(0.3)

# Completion message
progress_bar.progress(100, text="✅ Simulation Complete!")
st.balloons()

