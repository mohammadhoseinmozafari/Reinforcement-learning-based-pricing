import sys
import os
import time
import pandas as pd
import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from env.target_system import HotellingMarket

# Page config
st.set_page_config(
    page_title="Market Simulation",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main header with gradient */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3F9AAE 0%, #667eea 50%, #F96E5B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        letter-spacing: -1px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50;
        padding: 0.8rem 0;
        border-bottom: 3px solid #3F9AAE;
        margin-bottom: 1rem;
    }
    
    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #3F9AAE;
        transition: transform 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    /* Firm A specific styling */
    .firm-a-metric div[data-testid="stMetric"] {
        border-left-color: #3F9AAE;
    }
    
    /* Firm B specific styling */  
    .firm-b-metric div[data-testid="stMetric"] {
        border-left-color: #F96E5B;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Progress bar custom */
    .stProgress > div > div {
        background: white;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #f0f7fa 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #3F9AAE;
    }
    
    .info-box-orange {
        background: linear-gradient(135deg, #fef3e8 0%, #fdf8f4 100%);
        border-left-color: #F96E5B;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

def moving_average(series, window_size=10):
    return series.rolling(window=window_size).mean()

def create_pricing_inputs(column, firm_name, regime, default_uniform=2.0, default_new=1.5, default_old=2.5):
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

# Color palette
COLORS = {
    'firm_a': '#3F9AAE',
    'firm_b': '#F96E5B',
    'accent': '#667eea',
    'success': '#28a745',
    'warning': '#ffc107',
    'new_customers': '#9b59b6',
    'old_customers': '#2ecc71',
}

# Main UI
st.markdown('<h1 class="main-header">Dynamic Pricing Market Simulation</h1>', unsafe_allow_html=True)

# Settings in sidebar
with st.sidebar:
    st.markdown("## ⚙️ Simulation Settings")
    
    # Simulation controls
    st.markdown("### ⏱️ Simulation")
    n_steps = st.slider("Number of Steps", min_value=10, max_value=200, value=100, step=10)
    speed = st.select_slider("Speed", options=["Slow", "Normal", "Fast", "Instant"], value="Normal")
    speed_map = {"Slow": 0.5, "Normal": 0.3, "Fast": 0.1, "Instant": 0.01}
    
    st.markdown("---")
    
    st.markdown("### 🏢 Firm A")
    regime_a = st.segmented_control(
        label="Pricing Strategy",
        options=REGIME_OPTIONS.keys(),
        default=0,
        format_func=lambda x: REGIME_OPTIONS[x],
        selection_mode="single",
        key="regime_a"
    )
    prices_a = create_pricing_inputs(st, "Firm A", regime_a)
    
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
    prices_b = create_pricing_inputs(st, "Firm B", regime_b)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Market Parameters")
    n_consumers = st.number_input(
        label="👥 Number of Consumers",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Total consumers in the market"
    )
    
    exclusivity = st.slider(
        label="✨ Exclusivity Seekness (α)", 
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="How much consumers prefer less popular firms"
    )
    
    strategicness = st.slider(
        label="🧠 Strategicness (β)", 
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="How much consumers consider future utility"
    )

# Initialize market
market = HotellingMarket(n_consumers, 42, exclusivity, strategicness)
market.set_regimes(regime_a, regime_b)

# Get initial consumer stats
consumer_stats = market.get_consumer_stats()

# ==================== TOP METRICS ROW ====================
st.markdown('<p class="section-header">📈 Real-Time Performance Metrics</p>', unsafe_allow_html=True)

metric_row1 = st.columns(6)
m_share_a = metric_row1[0].empty()
m_share_b = metric_row1[1].empty()
m_profit_a = metric_row1[2].empty()
m_profit_b = metric_row1[3].empty()
m_retention_a = metric_row1[4].empty()
m_retention_b = metric_row1[5].empty()

metric_row2 = st.columns(6)
m_demand_new_a = metric_row2[0].empty()
m_demand_old_a = metric_row2[1].empty()
m_demand_new_b = metric_row2[2].empty()
m_demand_old_b = metric_row2[3].empty()
m_hhi = metric_row2[4].empty()
m_period = metric_row2[5].empty()

st.markdown("---")

# ==================== CHARTS SECTION ====================
tab_main, tab_customer, tab_analysis = st.tabs(["📊 Main Dashboard", "👥 Customer Insights", "📈 Deep Analysis"])

with tab_main:
    # Main charts - 2x2 grid
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### 📊 Market Share Over Time")
        market_share_data = pd.DataFrame({"Firm A": [], "Firm B": []})
        chart_share = st.line_chart(market_share_data, color=[COLORS['firm_a'], COLORS['firm_b']], height=280)
        
        st.markdown("#### 💵 Period Profits")
        profit_data = pd.DataFrame({"Firm A": [], "Firm B": []})
        chart_profit = st.line_chart(profit_data, color=[COLORS['firm_a'], COLORS['firm_b']], height=280)
    
    with chart_col2:
        st.markdown("#### 📉 Market Share (10-Period Moving Avg)")
        moving_avg_data = pd.DataFrame({"Firm A": [], "Firm B": []})
        chart_avg = st.line_chart(moving_avg_data, color=[COLORS['firm_a'], COLORS['firm_b']], height=280)
        
        st.markdown("#### 💰 Cumulative Profits")
        cumulative_data = pd.DataFrame({"Firm A": [], "Firm B": []})
        chart_cumulative = st.line_chart(cumulative_data, color=[COLORS['firm_a'], COLORS['firm_b']], height=280)

with tab_customer:
    cust_col1, cust_col2 = st.columns(2)
    
    with cust_col1:
        st.markdown("#### 🆕 New vs Returning Customers - Firm A")
        new_old_a_data = pd.DataFrame({"New Customers": [], "Returning Customers": []})
        chart_new_old_a = st.area_chart(new_old_a_data, color=[COLORS['new_customers'], COLORS['old_customers']], height=280)
        
        st.markdown("#### 🔄 Customer Retention Rate")
        retention_data = pd.DataFrame({"Firm A": [], "Firm B": []})
        chart_retention = st.line_chart(retention_data, color=[COLORS['firm_a'], COLORS['firm_b']], height=280)
    
    with cust_col2:
        st.markdown("#### 🆕 New vs Returning Customers - Firm B")
        new_old_b_data = pd.DataFrame({"New Customers": [], "Returning Customers": []})
        chart_new_old_b = st.area_chart(new_old_b_data, color=[COLORS['new_customers'], COLORS['old_customers']], height=280)
        
        st.markdown("#### 📊 New/Old Customer Ratio")
        ratio_data = pd.DataFrame({"Firm A Ratio": [], "Firm B Ratio": []})
        chart_ratio = st.line_chart(ratio_data, color=[COLORS['firm_a'], COLORS['firm_b']], height=280)

with tab_analysis:
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.markdown("#### 🏛️ Market Concentration (HHI)")
        hhi_data = pd.DataFrame({"HHI": []})
        chart_hhi = st.area_chart(hhi_data, color=[COLORS['accent']], height=280)
        
        st.markdown("#### ⚖️ Relative Popularity (Firm A / Firm B)")
        rel_pop_data = pd.DataFrame({"Relative Popularity": []})
        chart_rel_pop = st.line_chart(rel_pop_data, color=[COLORS['accent']], height=280)
    
    with analysis_col2:
        st.markdown("#### 📈 Profit Trend (5-Period)")
        profit_trend_data = pd.DataFrame({"Firm A Trend": [], "Firm B Trend": []})
        chart_profit_trend = st.line_chart(profit_trend_data, color=[COLORS['firm_a'], COLORS['firm_b']], height=280)
        
        st.markdown("#### 📊 Market Share Delta")
        delta_data = pd.DataFrame({"Share Change": []})
        chart_delta = st.bar_chart(delta_data, color=[COLORS['accent']], height=280)

# Consumer distribution info
with st.expander("📊 Consumer Population Statistics", expanded=False):
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.metric("Mean α (Exclusivity)", f"{consumer_stats['mean_alpha']:.3f}")
        st.metric("Std α", f"{consumer_stats['std_alpha']:.3f}")
    
    with stat_col2:
        st.metric("Mean β (Strategicness)", f"{consumer_stats['mean_beta']:.3f}")
        st.metric("Std β", f"{consumer_stats['std_beta']:.3f}")
    
    with stat_col3:
        st.metric("Mean Location", f"{consumer_stats['mean_location']:.3f}")
        st.metric("Std Location", f"{consumer_stats['std_location']:.3f}")

# Progress bar
st.markdown("---")
progress_bar = st.progress(0, text="🚀 Initializing simulation...")

# ==================== RUN SIMULATION ====================
prev_share_a = 0.5
prev_share_b = 0.5

for step in range(n_steps):
    market.step(prices_a, prices_b)
    
    # Get firm states
    state_a = market.get_firm_state(0)
    state_b = market.get_firm_state(1)
    
    # Calculate values
    firm_a_avg = moving_average(pd.Series(market.firms[0].market_share_history))
    firm_b_avg = moving_average(pd.Series(market.firms[1].market_share_history))
    
    current_share_a = state_a['market_share']
    current_share_b = state_b['market_share']
    current_profit_a = state_a['period_profit']
    current_profit_b = state_b['period_profit']
    total_profit_a = state_a['cumulative_profit']
    total_profit_b = state_b['cumulative_profit']
    retention_a = state_a['retention_rate']
    retention_b = state_b['retention_rate']
    demand_new_a = state_a['demand_new']
    demand_old_a = state_a['demand_old']
    demand_new_b = state_b['demand_new']
    demand_old_b = state_b['demand_old']
    hhi = market.get_market_concentration()
    new_old_ratio_a = state_a['new_old_ratio']
    new_old_ratio_b = state_b['new_old_ratio']
    profit_trend_a = state_a['profit_trend']
    profit_trend_b = state_b['profit_trend']
    rel_pop_a = state_a['relative_popularity']
    
    # ========== UPDATE METRICS ==========
    share_delta_a = current_share_a - prev_share_a
    share_delta_b = current_share_b - prev_share_b
    
    m_share_a.metric("🏢 Firm A Share", f"{current_share_a:.1%}", 
                     delta=f"{share_delta_a:+.1%}" if step > 0 else None)
    m_share_b.metric("🏭 Firm B Share", f"{current_share_b:.1%}",
                     delta=f"{share_delta_b:+.1%}" if step > 0 else None)
    m_profit_a.metric("💵 Firm A Profit", f"${total_profit_a:,.0f}",
                      delta=f"${current_profit_a:,.0f}/period")
    m_profit_b.metric("💰 Firm B Profit", f"${total_profit_b:,.0f}",
                      delta=f"${current_profit_b:,.0f}/period")
    m_retention_a.metric("🔄 Firm A Retention", f"{retention_a:.1%}")
    m_retention_b.metric("🔄 Firm B Retention", f"{retention_b:.1%}")
    
    m_demand_new_a.metric("🆕 A New Customers", f"{demand_new_a:,}")
    m_demand_old_a.metric("👥 A Returning", f"{demand_old_a:,}")
    m_demand_new_b.metric("🆕 B New Customers", f"{demand_new_b:,}")
    m_demand_old_b.metric("👥 B Returning", f"{demand_old_b:,}")
    m_hhi.metric("📊 HHI Index", f"{hhi:.3f}", 
                 help="0.5 = Equal split, 1.0 = Monopoly")
    m_period.metric("⏱️ Period", f"{step + 1}/{n_steps}")
    
    # ========== UPDATE MAIN CHARTS ==========
    chart_share.add_rows(pd.DataFrame({"Firm A": [current_share_a], "Firm B": [current_share_b]}))
    chart_avg.add_rows(pd.DataFrame({"Firm A": [firm_a_avg.iloc[-1] if not pd.isna(firm_a_avg.iloc[-1]) else current_share_a], 
                                      "Firm B": [firm_b_avg.iloc[-1] if not pd.isna(firm_b_avg.iloc[-1]) else current_share_b]}))
    chart_profit.add_rows(pd.DataFrame({"Firm A": [current_profit_a], "Firm B": [current_profit_b]}))
    chart_cumulative.add_rows(pd.DataFrame({"Firm A": [total_profit_a], "Firm B": [total_profit_b]}))
    
    # ========== UPDATE CUSTOMER CHARTS ==========
    chart_new_old_a.add_rows(pd.DataFrame({"New Customers": [demand_new_a], "Returning Customers": [demand_old_a]}))
    chart_new_old_b.add_rows(pd.DataFrame({"New Customers": [demand_new_b], "Returning Customers": [demand_old_b]}))
    chart_retention.add_rows(pd.DataFrame({"Firm A": [retention_a], "Firm B": [retention_b]}))
    chart_ratio.add_rows(pd.DataFrame({"Firm A Ratio": [new_old_ratio_a], "Firm B Ratio": [new_old_ratio_b]}))
    
    # ========== UPDATE ANALYSIS CHARTS ==========
    chart_hhi.add_rows(pd.DataFrame({"HHI": [hhi]}))
    chart_rel_pop.add_rows(pd.DataFrame({"Relative Popularity": [rel_pop_a]}))
    chart_profit_trend.add_rows(pd.DataFrame({"Firm A Trend": [profit_trend_a], "Firm B Trend": [profit_trend_b]}))
    chart_delta.add_rows(pd.DataFrame({"Share Change": [share_delta_a]}))
    
    # Update previous values
    prev_share_a = current_share_a
    prev_share_b = current_share_b
    
    # Update progress
    progress_bar.progress((step + 1) / n_steps, text=f"⏳ Simulation Progress: Step {step + 1}/{n_steps}")
    
    time.sleep(speed_map[speed])

# ==================== COMPLETION ====================
progress_bar.progress(100, text="✅ Simulation Complete!")

# Final summary
st.markdown("---")
st.markdown('<p class="section-header">📋 Final Summary</p>', unsafe_allow_html=True)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown("### 🏢 Firm A Results")
    st.metric("Final Market Share", f"{current_share_a:.1%}")
    st.metric("Total Profit", f"${total_profit_a:,.2f}")
    st.metric("Avg Retention Rate", f"{np.mean(market.firms[0].retention_history[-20:]) if market.firms[0].retention_history else 0:.1%}")
    st.metric("Strategy", REGIME_OPTIONS[regime_a])

with summary_col2:
    st.markdown("### 🏭 Firm B Results")
    st.metric("Final Market Share", f"{current_share_b:.1%}")
    st.metric("Total Profit", f"${total_profit_b:,.2f}")
    st.metric("Avg Retention Rate", f"{np.mean(market.firms[1].retention_history[-20:]) if market.firms[1].retention_history else 0:.1%}")
    st.metric("Strategy", REGIME_OPTIONS[regime_b])

with summary_col3:
    st.markdown("### 📊 Market Analysis")
    winner = "Firm A" if total_profit_a > total_profit_b else "Firm B" if total_profit_b > total_profit_a else "Tie"
    profit_diff = abs(total_profit_a - total_profit_b)
    st.metric("🏆 Winner (Profit)", winner)
    st.metric("Profit Difference", f"${profit_diff:,.2f}")
    st.metric("Final HHI", f"{hhi:.3f}")
    st.metric("Market Type", "Competitive" if hhi < 0.4 else "Moderate" if hhi < 0.6 else "Concentrated")

st.balloons()
