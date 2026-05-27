import pandas as pd
import streamlit as st
import json
st.set_page_config(
    page_title="Uniform Pricing Classic Optimization",
    page_icon="📊",
    layout="wide",
)

colors = {
    'soft_blue':    "#A8CEEF",
    'harsh_blue':    "#44A1F3",    # iPad Air blue
    'soft_green':   "#7DF67D",   # iPhone 15 green
    'soft_orange':  "#FEB193",   # iMac orange (pastel)
    'soft_pink':    "#F5C5D5",
    'harsh_pink':    "#F58AAD",   # iPhone 15 pink
    'soft_purple':  '#D5C5F0',   # iPad Air purple
    'harsh_purple': "#BD6DFB",
    'soft_teal':    "#BDEEE2",   # iMac teal (pastel)
    'harsh_teal':    "#0BD4A2",   # iMac teal (pastel)
    'soft_gray':    "#C9C9E1",   # starlight
    'soft_yellow':  '#FFF5C0',   # iPhone 14 yellow
    'soft_red':     "#F68C9D",   # PRODUCT(RED) pastel
}
st.markdown("""
<style>
    /* Reset & global */
    .main {
        background: linear-gradient(180deg, #F9F9F9 0%, #FFFFFF 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Typography */
    .apple-title {
        font-size: 34px;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .apple-subtitle {
        font-size: 15px;
        font-weight: 400;
        color: #86868B;
        margin-bottom: 28px;
    }

    /* Metric cards – floating, airy */
    .metric-card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 18px 22px;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 500;
        color: #8E8E93;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .metric-trend {
        font-size: 13px;
        font-weight: 500;
        color: #34C759;  /* system green for positive */
        margin-top: 8px;
    }
    .metric-trend.negative {
        color: #FF3B30;
    }

    /* Divider */
    .apple-divider {
        height: 1px;
        background: #E5E5EA;
        margin: 28px 0;
    }

    /* Sidebar – soft minimal */
    [data-testid="stSidebar"] {
        background-color: #FBFBFD;
        border-right: 1px solid #E5E5EA;
    }

    /* Tabs – Apple’s underline style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 28px;
        background: transparent;
        border-bottom: 1px solid #E5E5EA;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8E8E93;
        font-weight: 500;
        font-size: 15px;
        padding: 10px 0;
        margin-right: 0;
        transition: color 0.15s;
    }
    .stTabs [aria-selected="true"] {
        color: #1D1D1F;
        border-bottom: 2px solid #5D9BEC;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

data = load_data("optimization/uniform_pricing/aggressive_uniform")

st.title("Uniform Pricing Classic Optimization Results")

metrics = pd.DataFrame({
    'prices': data['prices'],
    'opponent_prices': data['opponent_prices'],
    'profits': data['profits'],
    'market_shares': data['market_shares'],
    'demands':  data['demands']
})



