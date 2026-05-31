
import numpy as np
import pandas as pd
import streamlit as st
import json
import plotly.graph_objects as go
st.set_page_config(
    page_title="Uniform Pricing Classic Optimization",
    page_icon="📊",
    layout="wide",
)

colors = {
    'soft_blue':    "#A8CEEF",
    'harsh_blue':    "#44A1F3",    # iPad Air blue
    'soft_green':   "#7DF67D",   # iPhone 15 green
    'soft_orange':  "#F4A88A",   # iMac orange (pastel)
    'harsh_orange':  "#F9652B",   # iMac orange (pastel)

    'soft_pink':    "#F5C5D5",
    'harsh_pink':    "#F58AAD",   # iPhone 15 pink
    'soft_purple':  '#D5C5F0',   # iPad Air purple
    'harsh_purple': "#BD6DFB",
    'soft_teal':    "#BDEEE2",   # iMac teal (pastel)
    'harsh_teal':    "#0BD4A2",   # iMac teal (pastel)
    'soft_gray':    "#C9C9E1",   # starlight
    'soft_yellow':  '#FFF5C0',   # iPhone 14 yellow
    'soft_red':     "#F68C9D",   # PRODUCT(RED) pastel
    'harsh_gray':    "#212125"

}
st.markdown("""
<style>
    /* Reset & global */
    .main {
        background: linear-gradient(180deg, #F9F9F9 0%, #FFFFFF 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Typography */
    .title {
        font-size: 34px;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .subtitle {
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
        font-size: 13px;
        font-weight: 700;
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
    .divider {
        height: 1px;
        background: #E5E5EA;
        margin: 28px 0;
    }

   
   
</style>
""", unsafe_allow_html=True)

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


st.sidebar.header("Optimization Settings")
options = {
    "aggressive_uniform": "Aggressive Uniform",
    "passive_uniform": "Passive Uniform",
    "premium_uniform" : "Premium Uniform"
}

opponent_type = st.sidebar.selectbox(
    "Opponent Type",
    options=options.keys(),
    index=0,
)

data = load_data(f"optimization/uniform_pricing/results/{opponent_type}/optimization_{opponent_type}.json")


metrics = pd.DataFrame({
    'prices': data['prices'],
    'profits': data['profits'],
    'market_shares': data['market_shares'],
    'demands':  data['demands']
})
optimal_idx = data['optimal_idx']

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown('<div class="title">Uniform Pricing Classic Optimization</div>', unsafe_allow_html=True)

col1 , col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label" style="color:#{colors['harsh_blue'][1:]};">Opponent Type</div>
        <div class="metric-value">{options[opponent_type]}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label" style="color:#{colors['harsh_teal'][1:]};">Optimal Price</div>
        <div class="metric-value">{metrics['prices'][optimal_idx]:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    optimal_ms = metrics['market_shares'][optimal_idx]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label" style="color:#{colors['soft_orange'][1:]};">Optimal Market Share</div>
        <div class="metric-value">{optimal_ms:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    opt_profit= metrics['profits'][optimal_idx]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label" style="color:#{colors['harsh_purple'][1:]};">Optimal Profit</div>
        <div class="metric-value">{opt_profit:.1f}</div>
      
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Profit vs Price")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x= metrics['prices'],
            y = metrics['profits'],
            
            name = "Price vs Profit",
            mode='lines',
            line=dict(color=colors['harsh_purple'], width=3),
            opacity=1.0
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[metrics['prices'][optimal_idx]],
            y=[metrics['profits'][optimal_idx]],
            mode="markers+text",
            name=f"Optimal: p*={metrics['prices'][optimal_idx]:.2f}",
            marker=dict(color=colors['harsh_gray'], size=12, symbol="circle"),
            text=[f"π*={metrics['profits'][optimal_idx]:.1f}"],
            textposition="top center",
            textfont=dict(size=13, color=colors['harsh_gray']),
        )
    )
    fig.add_trace(
            go.Scatter(
                x=np.concatenate([metrics['prices'], metrics['prices'][::-1]]),
                y=np.concatenate([metrics['profits'], [metrics['profits'].min()]*len(metrics)]),
                name = "Area",
                showlegend=False,
                fillcolor=colors['soft_purple'],
                line=dict(
                    width=0
                ),
                fill='toself',
                opacity=0.2

            )
        )
    fig.add_vline(
        x=metrics['prices'][optimal_idx],
        line_dash="dash",
        line_color=colors['harsh_gray'],
        opacity=0.5,
    )
    fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
        
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            xaxis=dict(
                title='Price',
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title='Profit',
                showgrid=True,
                gridcolor='#E5E5EA',
                zeroline=False,
                tickfont=dict(size=14),
                
            ),
            hovermode='x unified',

        )
    
    st.plotly_chart(fig)
with col2:
    st.markdown("#### Market Share vs Price")    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x= metrics['prices'],
            y = metrics['market_shares'],
            
            name = "Market Share vs Price",
            mode='lines',
            line=dict(color=colors['harsh_orange'], width=3),
            opacity=1.0
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[metrics['prices'][optimal_idx]],
            y=[metrics['market_shares'][optimal_idx]],
            mode="markers+text",
            name=f"Optimal: p*={metrics['prices'][optimal_idx]:.2f}",
            marker=dict(color=colors['harsh_gray'], size=12, symbol="circle"),
            text=[f"d*={metrics['market_shares'][optimal_idx]:.1f}"],
            textposition="top center",
            textfont=dict(size=13, color=colors['harsh_gray']),
        )
    )
    fig.add_trace(
            go.Scatter(
                x=np.concatenate([metrics['prices'], metrics['prices'][::-1]]),
                y=np.concatenate([metrics['market_shares'], [metrics['market_shares'].min()]*len(metrics)]),
                name = "Area",
                showlegend=False,
                fillcolor=colors['soft_orange'],
                line=dict(
                    width=0
                ),
                fill='toself',
                opacity=0.2

            )
        )
    
    
    fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
        
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            xaxis=dict(
                title='Price',
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title='Market Share',
                showgrid=True,
                gridcolor='#E5E5EA',
                zeroline=False,
                tickfont=dict(size=14),
                
            ),
            hovermode='x unified',

        )
    
    st.plotly_chart(fig)