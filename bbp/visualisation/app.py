import streamlit as st

st.set_page_config(
    page_title="A Reinforcement Learning approach to pricing strategies",
    page_icon="🤖",
    layout="wide",
)

from utils.styles import apply_styles
apply_styles()

# ── Hero section ──────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🤖 RL Dashboard Suite</div>
    <div class="hero-subtitle">A Reinforcement Learning approach to pricing strategies</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)

# ── Dashboard cards (3 columns) ──────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <a href="/Training" style="text-decoration: none; color: inherit; display: block;">
        <div class="dashboard-card">
            <div class="card-hover-effect"></div>
            <span class="card-badge">LIVE</span>
            <span class="card-icon">📊</span>
            <div class="card-title">Training Dashboard</div>
            <div class="card-description">
                Monitor training progress, view reward curves, 
                and track agent performance in real-time
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <a href="/Evaluation" style="text-decoration: none; color: inherit; display: block;">
        <div class="dashboard-card">
            <div class="card-hover-effect"></div>
            <span class="card-badge">ANALYTICS</span>
            <span class="card-icon">📈</span>
            <div class="card-title">Evaluation Dashboard</div>
            <div class="card-description">
                Comprehensive evaluation metrics, success rates,
                and statistical analysis of agent performance
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <a href="/Simulation" style="text-decoration: none; color: inherit; display: block;">
        <div class="dashboard-card">
            <div class="card-hover-effect"></div>
            <span class="card-badge">INTERACTIVE</span>
            <span class="card-icon">🎮</span>
            <div class="card-title">Simulation Dashboard</div>
            <div class="card-description">
                Test your trained agent in interactive simulations
                and compare against baseline strategies
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)