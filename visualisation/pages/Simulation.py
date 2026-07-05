"""Interactive Streamlit game: a human prices against a trained SAC agent."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import (
    PRICE_BBP_NEW_MAX,
    PRICE_BBP_NEW_MIN,
    PRICE_BBP_OLD_MAX,
    PRICE_BBP_OLD_MIN,
    PRICE_UNIFORM_MAX,
    PRICE_UNIFORM_MIN,
)
from visualisation.utils.simulation_game import SimulationGame, create_game
from visualisation.utils.styles import apply_styles, colors, metric_card


st.set_page_config(page_title="Play Against the Agent", page_icon="🎮", layout="wide")
apply_styles()

STRATEGY_LABELS = {
    "uniform": "Uniform Pricing",
    "bbp": "Behavior-Based Pricing",
}
GAME_KEY = "human_vs_agent_game"


def chart_layout(fig: go.Figure, y_title: str) -> go.Figure:
    """Apply the same clean Plotly layout used by the other dashboards."""
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        xaxis=dict(title="Step", showgrid=False, zeroline=False),
        yaxis=dict(title=y_title, showgrid=True, gridcolor="#E5E5EA", zeroline=False),
        hovermode="x unified",
    )
    return fig


def line_chart(data: pd.DataFrame, series: list[tuple[str, str, str, str]], y_title: str) -> go.Figure:
    fig = go.Figure()
    for column, label, color, dash in series:
        fig.add_trace(go.Scatter(
            x=data["step"],
            y=data[column],
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=5),
        ))
    return chart_layout(fig, y_title)


def price_series(game: SimulationGame) -> list[tuple[str, str, str, str]]:
    series: list[tuple[str, str, str, str]] = []
    if game.human_strategy == "uniform":
        series.append(("user_uniform_price", "Your Uniform Price", colors["harsh_teal"], "solid"))
    else:
        series.extend([
            ("user_new_price", "Your New-Customer Price", colors["harsh_teal"], "solid"),
            ("user_old_price", "Your Returning-Customer Price", colors["harsh_blue"], "solid"),
        ])
    if game.agent_strategy == "uniform":
        series.append(("agent_uniform_price", "Agent Uniform Price", colors["harsh_pink"], "dot"))
    else:
        series.extend([
            ("agent_new_price", "Agent New-Customer Price", colors["harsh_pink"], "dot"),
            ("agent_old_price", "Agent Returning-Customer Price", colors["harsh_purple"], "dot"),
        ])
    return series


with st.sidebar:
    st.markdown("### Game Setup")
    agent_strategy = st.radio(
        "Agent pricing strategy",
        options=list(STRATEGY_LABELS),
        format_func=STRATEGY_LABELS.get,
        horizontal=False,
    )
    human_strategy = st.radio(
        "Your pricing strategy",
        options=list(STRATEGY_LABELS),
        format_func=STRATEGY_LABELS.get,
        horizontal=False,
    )
    st.caption(
        f"Loads `experiments/{agent_strategy}_vs_{human_strategy}/runs/1/`"
    )
    start_game = st.button("Start / Restart Episode", type="primary", width="stretch")


if start_game:
    old_game = st.session_state.get(GAME_KEY)
    if old_game is not None:
        old_game.env.close()
    try:
        with st.spinner("Loading trained agent and starting the market..."):
            st.session_state[GAME_KEY] = create_game(
                PROJECT_ROOT, agent_strategy, human_strategy
            )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.session_state.pop(GAME_KEY, None)
        st.error(str(exc))


st.markdown('<div class="hero-title">Play Against the Pricing Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Post a price, let the trained agent respond, and compete for the market.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)

game: SimulationGame | None = st.session_state.get(GAME_KEY)
if game is None:
    st.info("Choose both pricing strategies in the sidebar, then start an episode.")
    st.stop()

selection_changed = (
    game.agent_strategy != agent_strategy or game.human_strategy != human_strategy
)
if selection_changed:
    st.warning("The strategy selection changed. Start a new episode to load the matching agent.")

setup_cols = st.columns(3)
setup_cols[0].markdown(metric_card("Trained Agent", STRATEGY_LABELS[game.agent_strategy]), unsafe_allow_html=True)
setup_cols[1].markdown(metric_card("Your Strategy", STRATEGY_LABELS[game.human_strategy]), unsafe_allow_html=True)
setup_cols[2].markdown(
    metric_card("Episode Progress", f"{game.step_number} / {game.episode_length}"),
    unsafe_allow_html=True,
)

control_col, response_col = st.columns([1, 1])
with control_col:
    st.markdown("### Post Your Price")
    with st.form("human_price_form", clear_on_submit=False):
        if game.human_strategy == "uniform":
            uniform_price = st.number_input(
                "Uniform price",
                min_value=float(PRICE_UNIFORM_MIN),
                max_value=float(PRICE_UNIFORM_MAX),
                value=2.5,
                step=0.1,
            )
            price_new, price_old = 2.0, 3.0
        else:
            uniform_price = 2.5
            price_new = st.number_input(
                "New-customer price",
                min_value=float(PRICE_BBP_NEW_MIN),
                max_value=float(PRICE_BBP_NEW_MAX),
                value=2.0,
                step=0.1,
            )
            price_old = st.number_input(
                "Returning-customer price",
                min_value=max(float(PRICE_BBP_OLD_MIN), float(price_new)),
                max_value=float(PRICE_BBP_OLD_MAX),
                value=max(3.0, float(price_new)),
                step=0.1,
            )
        submitted = st.form_submit_button(
            "Post Price and Play Step",
            type="primary",
            width="stretch",
            disabled=game.finished or selection_changed,
        )

    if submitted:
        try:
            game.step(float(uniform_price), float(price_new), float(price_old))
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.rerun()

with response_col:
    st.markdown("### Latest Market Response")
    if not game.history:
        st.caption("The agent response and market result will appear after your first posted price.")
    else:
        latest = game.history[-1]
        if game.agent_strategy == "uniform":
            agent_price_text = f"{latest['agent_uniform_price']:.2f}"
        else:
            agent_price_text = f"{latest['agent_new_price']:.2f} / {latest['agent_old_price']:.2f}"
        response_metrics = st.columns(2)
        response_metrics[0].metric("Agent posted price", agent_price_text)
        response_metrics[1].metric("Your market share", f"{latest['user_market_share']:.1%}")
        response_metrics[0].metric("Your period profit", f"{latest['user_profit']:.2f}")
        response_metrics[1].metric("Agent period profit", f"{latest['agent_profit']:.2f}")

if game.finished:
    user_total = game.history[-1]["user_cumulative_profit"]
    agent_total = game.history[-1]["agent_cumulative_profit"]
    if user_total > agent_total:
        st.success(f"Episode complete — you won by {user_total - agent_total:.2f} profit units.")
    elif user_total < agent_total:
        st.warning(f"Episode complete — the agent won by {agent_total - user_total:.2f} profit units.")
    else:
        st.info("Episode complete — the market ended in a tie.")

if game.history:
    frame = pd.DataFrame(game.history)
    st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)
    tab_market, tab_prices, tab_data = st.tabs(["Market Results", "Pricing", "Step Data"])

    with tab_market:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Period Profit")
            st.plotly_chart(line_chart(frame, [
                ("user_profit", "Your Profit", colors["harsh_teal"], "solid"),
                ("agent_profit", "Agent Profit", colors["harsh_pink"], "solid"),
            ], "Profit"), width="stretch")
            st.markdown("#### Market Share")
            share_frame = frame.copy()
            share_frame["user_market_share"] *= 100
            share_frame["agent_market_share"] *= 100
            st.plotly_chart(line_chart(share_frame, [
                ("user_market_share", "Your Market Share", colors["harsh_teal"], "solid"),
                ("agent_market_share", "Agent Market Share", colors["harsh_pink"], "solid"),
            ], "Market Share (%)"), width="stretch")
        with right:
            st.markdown("#### Cumulative Profit")
            st.plotly_chart(line_chart(frame, [
                ("user_cumulative_profit", "Your Cumulative Profit", colors["harsh_teal"], "solid"),
                ("agent_cumulative_profit", "Agent Cumulative Profit", colors["harsh_pink"], "solid"),
            ], "Cumulative Profit"), width="stretch")
            st.markdown("#### Profit Advantage")
            gap_frame = frame.copy()
            gap_frame["profit_gap"] = (
                gap_frame["user_cumulative_profit"] - gap_frame["agent_cumulative_profit"]
            )
            gap_figure = line_chart(gap_frame, [
                ("profit_gap", "Your Advantage", colors["harsh_purple"], "solid"),
            ], "Cumulative Profit Gap")
            gap_figure.add_hline(y=0, line_dash="dash", line_color=colors["harsh_gray"])
            st.plotly_chart(gap_figure, width="stretch")

    with tab_prices:
        st.markdown("#### Posted Prices")
        st.plotly_chart(
            line_chart(frame, price_series(game), "Price"),
            width="stretch",
        )

    with tab_data:
        st.dataframe(frame, width="stretch", hide_index=True)
        st.download_button(
            "Download Episode CSV",
            data=frame.to_csv(index=False),
            file_name=f"{game.agent_strategy}_vs_{game.human_strategy}_episode.csv",
            mime="text/csv",
        )
