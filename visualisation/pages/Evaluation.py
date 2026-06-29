import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd

from utils import load_data, PathResolver, moving_average, apply_styles, colors


apply_styles()
window = 10


EVAL_SCENARIOS = {
	"Passive BBP": "eval_result_passive_bbp.json",
	"Passive Aggressive BBP": "eval_result_passive_aggressive_bbp.json",
	"Premium BBP": "eval_result_premium_bbp.json",
	"Premium Passive BBP": "eval_result_premium_passive_bbp.json",
	"Aggressive BBP": "eval_result_aggressive_bbp.json",
}


def build_layout(fig: go.Figure, height: int = 350) -> go.Figure:
	fig.update_layout(
		plot_bgcolor="white",
		paper_bgcolor="white",
		height=height,
		margin=dict(l=0, r=0, t=0, b=0),
		legend=dict(
			orientation="h",
			yanchor="bottom",
			y=1.02,
			xanchor="right",
			x=1,
			font=dict(size=12),
		),
		xaxis=dict(
			showgrid=False,
			zeroline=False,
			tickfont=dict(size=12),
		),
		yaxis=dict(
			showgrid=True,
			gridcolor="#E5E5EA",
			zeroline=False,
			tickfont=dict(size=12),
		),
		hovermode="x unified",
	)
	return fig


def pad_moving_average(series, window_size, total_length):
	averaged = np.asarray(moving_average(series, window=window_size), dtype=float)
	if len(averaged) < total_length:
		averaged = np.concatenate([np.repeat(np.nan, total_length - len(averaged)), averaged])
	return averaged


st.sidebar.markdown("### Settings")
evaluation_phase = st.sidebar.radio(
	"Evaluation Phase",
	["Uniform Pricing"],
	index=0,
)

evaluation_scenario = st.sidebar.radio(
	"Evaluation Scenario",
	list(EVAL_SCENARIOS.keys()),
	index=0,
)

path_resolver = PathResolver()
path = path_resolver.resolve_eval_path(
	phase=evaluation_phase,
	opponent_type="BBP Opponent",
)

data = load_data(f"{path}/{EVAL_SCENARIOS[evaluation_scenario]}")

step_profits = data.get("step_profits", [])
step_prices = data.get("step_prices", [])
step_market_shares = data.get("step_market_shares", [])
step_opp_profits = data.get("step_opp_profits", [])
step_opp_uniform_prices = data.get("step_opp_uniform_prices", [])
step_opp_new_prices = data.get("step_opp_new_prices", [])
step_opp_old_prices = data.get("step_opp_old_prices", [])
market_share_raw = np.asarray(step_market_shares, dtype=float)
market_share_pct = market_share_raw * 100.0

df_metrics = pd.DataFrame(
	{
		"step": range(len(step_profits)),
		"agent_profit": step_profits,
		"opponent_profit": step_opp_profits,
		"agent_price": step_prices,
		"opponent_price_uniform": step_opp_uniform_prices,
		"opponent_price_new": step_opp_new_prices,
		"opponent_price_old": step_opp_old_prices,
		"market_share": market_share_pct,
	}
)

df_metrics["profit_gap"] = df_metrics["agent_profit"] - df_metrics["opponent_profit"]
df_metrics["price_gap_uniform"] = df_metrics["agent_price"] - df_metrics["opponent_price_uniform"]
df_metrics["price_gap_new"] = df_metrics["agent_price"] - df_metrics["opponent_price_new"]
df_metrics["price_gap_old"] = df_metrics["agent_price"] - df_metrics["opponent_price_old"]

total_steps = len(step_profits)
total_agent_profit = float(np.sum(step_profits)) if step_profits else 0.0
total_opponent_profit = float(np.sum(step_opp_profits)) if step_opp_profits else 0.0
profit_gap = total_agent_profit - total_opponent_profit
profit_gap_pct = (profit_gap / abs(total_opponent_profit) * 100) if total_opponent_profit != 0 else 0.0
final_market_share = float(market_share_pct[-1]) if total_steps else 0.0
avg_market_share = float(np.mean(market_share_pct)) if total_steps else 0.0
avg_step_profit = float(np.mean(step_profits)) if step_profits else 0.0
avg_agent_price = float(np.mean(step_prices)) if step_prices else 0.0
price_volatility = float(np.std(step_prices)) if step_prices else 0.0

st.sidebar.markdown("### 🎛️ Visualization Controls")
smoothing = st.sidebar.slider("Smoothing window", 1, 30, 10)
window = smoothing

if st.sidebar.checkbox("Show raw data", value=False):
	st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)
	st.markdown("### 📋 Raw Evaluation Data")
	data_option = st.selectbox(
		"Select data to view",
		["Step Metrics", "Price Comparison", "Summary"],
	)
	if data_option == "Step Metrics":
		st.dataframe(df_metrics, use_container_width=True)
	elif data_option == "Price Comparison":
		st.dataframe(
			df_metrics[
				[
					"step",
					"agent_price",
					"opponent_price_uniform",
					"opponent_price_new",
					"opponent_price_old",
					"price_gap_uniform",
					"price_gap_new",
					"price_gap_old",
				]
			],
			use_container_width=True,
		)
	else:
		st.dataframe(
			pd.DataFrame(
				{
					"metric": [
						"scenario",
						"steps",
						"total_agent_profit",
						"total_opponent_profit",
						"profit_gap",
						"final_market_share",
						"avg_market_share",
						"avg_agent_price",
						"price_volatility",
					],
					"value": [
						evaluation_scenario,
						total_steps,
						total_agent_profit,
						total_opponent_profit,
						profit_gap,
						final_market_share,
						avg_market_share,
						avg_agent_price,
						price_volatility,
					],
				}
			),
			use_container_width=True,
		)

st.sidebar.markdown("### 💾 Export Data")
if st.sidebar.button("Export metrics as CSV"):
	csv = df_metrics.to_csv(index=False)
	st.sidebar.download_button(
		label="Download CSV",
		data=csv,
		file_name="rl_evaluation_metrics.csv",
		mime="text/csv",
	)

st.markdown('<div class="hero-title">Evaluation Dashboard</div>', unsafe_allow_html=True)
st.markdown(
	'<div class="hero-subtitle">Policy performance, profit distribution, and price behavior</div>',
	unsafe_allow_html=True,
)

col0, col1, col2, col3, col4 = st.columns(5)

with col0:
	st.markdown(
		f"""
	<div class="metric-card">
		<div class="metric-label" style="color:#{colors['harsh_purple'][1:]};">Evaluation Scenario</div>
		<div class="metric-value" style="font-size:20px;">{evaluation_scenario}</div>
	</div>
	""",
		unsafe_allow_html=True,
	)

with col1:
	st.markdown(
		f"""
	<div class="metric-card">
		<div class="metric-label" style="color:#{colors['soft_blue'][1:]};">Evaluation Steps</div>
		<div class="metric-value">{total_steps}</div>
		<div class="metric-trend">Episode horizon</div>
	</div>
	""",
		unsafe_allow_html=True,
	)

with col2:
	st.markdown(
		f"""
	<div class="metric-card">
		<div class="metric-label" style="color:#{colors['soft_green'][1:]};">Total Agent Profit</div>
		<div class="metric-value">{total_agent_profit:.1f}</div>
		<div class="metric-trend">Avg step profit {avg_step_profit:.1f}</div>
	</div>
	""",
		unsafe_allow_html=True,
	)

with col3:
	gap_class = "negative" if profit_gap < 0 else ""
	st.markdown(
		f"""
	<div class="metric-card">
		<div class="metric-label" style="color:#{colors['soft_orange'][1:]};">Profit Gap</div>
		<div class="metric-value">{profit_gap:+.1f}</div>
		<div class="metric-trend {gap_class}">{profit_gap_pct:+.1f}% vs opponent</div>
	</div>
	""",
		unsafe_allow_html=True,
	)

with col4:
	market_leader = final_market_share > 50
	st.markdown(
		f"""
	<div class="metric-card">
		<div class="metric-label" style="color:#{colors['soft_pink'][1:]};">Final Market Share</div>
		<div class="metric-value">{final_market_share:.1f}%</div>
		<div class="metric-trend" style="color:{colors['soft_green'] if market_leader else colors['soft_red']};">
			{'📈 Dominant' if market_leader else '📉 Behind'}
		</div>
	</div>
	""",
		unsafe_allow_html=True,
	)

st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
	"📈 Evaluation Trajectory",
	"💰 Profit Analysis",
	"🏷️ Price Competition",
	"📊 Summary",
])

with tab1:
	col1, col2 = st.columns(2)

	with col1:
		ma_profits = pad_moving_average(step_profits, window, len(df_metrics))
		fig = go.Figure()
		fig.add_trace(
			go.Scatter(
				x=df_metrics["step"],
				y=df_metrics["agent_profit"],
				name="Step Profit",
				mode="lines",
				line=dict(color=colors["harsh_blue"], width=3),
				opacity=0.75,
			)
		)
		fig.add_trace(
			go.Scatter(
				x=np.concatenate([df_metrics["step"], df_metrics["step"][::-1]]),
				y=np.concatenate([df_metrics["agent_profit"], [0.0] * len(df_metrics)]),
				name="Area",
				showlegend=False,
				fillcolor=colors["soft_blue"],
				line=dict(width=0),
				fill="toself",
				opacity=0.2,
			)
		)
		fig.add_trace(
			go.Scatter(
				x=df_metrics["step"],
				y=ma_profits,
				name=f"Moving Avg ({window})",
				mode="lines",
				line=dict(color=colors["soft_red"], width=3),
			)
		)
		fig.add_hline(y=0.0, line_dash="dash", opacity=0.7, line_color="black")
		build_layout(fig)
		st.markdown("#### Step Profit Dynamics")
		st.plotly_chart(fig)

	with col2:
		ma_share = pad_moving_average(step_market_shares, window, len(df_metrics))
		fig = go.Figure()
		fig.add_trace(
			go.Scatter(
				x=df_metrics["step"],
				y=df_metrics["market_share"],
				name="Market Share",
				mode="lines",
				line=dict(color=colors["harsh_teal"], width=3),
				opacity=1.0,
			)
		)
		fig.add_trace(
			go.Scatter(
				x=df_metrics["step"],
				y=df_metrics["market_share"],
				fill="tonexty",
				fillcolor="rgba(0, 255, 0, 0.2)",
				line=dict(width=0),
				showlegend=False,
				name="Above 50%",
			)
		)
		df_metrics["above_05"] = df_metrics["market_share"].apply(lambda x: max(x, 50.0))
		df_metrics["below_05"] = df_metrics["market_share"].apply(lambda x: min(x, 50.0))

		fig.add_trace(
			go.Scatter(
				x=np.concatenate([df_metrics["step"], df_metrics["step"][::-1]]),
				y=np.concatenate([df_metrics["below_05"], pd.Series([50.0] * len(df_metrics))]),
				fill="toself",
				fillcolor=colors["soft_red"],
				line=dict(width=0),
				showlegend=False,
				name="Below 50%",
				opacity=0.3,
			)
		)
		fig.add_trace(
			go.Scatter(
				x=df_metrics["step"],
				y=ma_share,
				name=f"Moving Avg ({window})",
				mode="lines",
				line=dict(color=colors["soft_purple"], width=3),
				opacity=0.9,
			)
		)
		fig.add_hline(y=50.0, line_dash="dash", opacity=0.7, line_color="black")
		build_layout(fig)
		st.markdown("#### Market Share Trajectory")
		st.plotly_chart(fig)

	with col1:
		st.markdown("#### Profit Comparison")
		fig = go.Figure()
		fig.add_trace(
			go.Scatter(
				x=df_metrics["step"],
				y=df_metrics["agent_profit"],
				name="Agent Profit",
				mode="lines",
				line=dict(color=colors["harsh_teal"], width=2),
			)
		)
		fig.add_trace(
			go.Scatter(
				x=df_metrics["step"],
				y=df_metrics["opponent_profit"],
				name="Opponent Profit",
				mode="lines",
				line=dict(color=colors["harsh_pink"], width=2),
			)
		)
		build_layout(fig)
		st.plotly_chart(fig)

	with col2:
		st.markdown("#### Profit Gap")
		fig = go.Figure()
		fig.add_trace(
			go.Scatter(
				x=df_metrics["step"],
				y=df_metrics["profit_gap"],
				name="Agent - Opponent",
				mode="lines",
				line=dict(color=colors["harsh_purple"], width=3),
			)
		)
		fig.add_hline(y=0.0, line_dash="dash", opacity=0.7, line_color="black")
		build_layout(fig)
		st.plotly_chart(fig)

with tab2:
	st.markdown("#### Cumulative Profit Comparison")
	cum_agent = np.cumsum(step_profits)
	cum_opponent = np.cumsum(step_opp_profits)
	fig = go.Figure()

	fig.add_trace(
		go.Scatter(
			x=df_metrics["step"],
			y=cum_agent,
			name="Cumulative Agent Profit",
			mode="lines",
			line=dict(color=colors["harsh_teal"], width=3),
			opacity=1.0,
		)
	)
	fig.add_trace(
		go.Scatter(
			x=df_metrics["step"],
			y=cum_opponent,
			name="Cumulative Opponent Profit",
			mode="lines",
			line=dict(color=colors["harsh_pink"], width=3),
			fill="tonexty",
			fillcolor="rgba(189, 238, 226, 0.3)",
			opacity=1.0,
		)
	)
	build_layout(fig)
	st.plotly_chart(fig)

	col1, col2 = st.columns(2)
	with col1:
		st.markdown("#### Evaluation Profit Summary")
		st.metric("Total Agent Profit", f"{total_agent_profit:.1f}")
		st.metric("Total Opponent Profit", f"{total_opponent_profit:.1f}")
		st.metric("Average Step Profit", f"{avg_step_profit:.1f}")
	with col2:
		st.markdown("#### Market Summary")
		st.metric("Final Market Share", f"{final_market_share:.1f}%")
		st.metric("Average Market Share", f"{avg_market_share:.1f}%")
		st.metric("Profit Gap", f"{profit_gap:+.1f}")

with tab3:
	st.markdown("#### Price Competition")
	fig = go.Figure()
	fig.add_trace(
		go.Scatter(
			x=df_metrics["step"],
			y=df_metrics["agent_price"],
			name="Agent Price",
			mode="lines",
			line=dict(color=colors["harsh_blue"], width=2),
		)
	)
	fig.add_trace(
		go.Scatter(
			x=df_metrics["step"],
			y=df_metrics["opponent_price_uniform"],
			name="Opponent Price (Uniform)",
			mode="lines",
			line=dict(color=colors["harsh_pink"], width=2),
		)
	)
	fig.add_trace(
		go.Scatter(
			x=df_metrics["step"],
			y=df_metrics["opponent_price_new"],
			name="Opponent Price (New)",
			mode="lines",
			line=dict(color=colors["harsh_purple"], width=2),
		)
	)
	fig.add_trace(
		go.Scatter(
			x=df_metrics["step"],
			y=df_metrics["opponent_price_old"],
			name="Opponent Price (Established)",
			mode="lines",
			line=dict(color=colors["harsh_teal"], width=2),
		)
	)
	build_layout(fig)
	st.plotly_chart(fig)

	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric("Average Agent Price", f"{avg_agent_price:.2f}")
	with col2:
		st.metric("Price Volatility", f"{price_volatility:.3f}")
	with col3:
		st.metric("Final Agent Price", f"{step_prices[-1]:.2f}" if step_prices else "0.00")

	st.markdown("#### Price Gaps")
	fig = go.Figure()
	fig.add_trace(
		go.Scatter(
			x=df_metrics["step"],
			y=df_metrics["price_gap_uniform"],
			name="Gap vs Uniform",
			mode="lines",
			line=dict(color=colors["harsh_blue"], width=2.5),
		)
	)
	fig.add_trace(
		go.Scatter(
			x=df_metrics["step"],
			y=df_metrics["price_gap_new"],
			name="Gap vs New",
			mode="lines",
			line=dict(color=colors["harsh_purple"], width=2.5),
		)
	)
	fig.add_trace(
		go.Scatter(
			x=df_metrics["step"],
			y=df_metrics["price_gap_old"],
			name="Gap vs Established",
			mode="lines",
			line=dict(color=colors["harsh_teal"], width=2.5),
		)
	)
	fig.add_hline(y=0.0, line_dash="dash", opacity=0.7, line_color="black")
	build_layout(fig)
	st.plotly_chart(fig)

with tab4:
	col1, col2 = st.columns(2)

	with col1:
		st.markdown("#### Step Metrics")
		st.dataframe(df_metrics, use_container_width=True)

	with col2:
		st.markdown("#### Evaluation Summary")
		summary_df = pd.DataFrame(
			{
				"metric": [
					"scenario",
					"steps",
					"total_agent_profit",
					"total_opponent_profit",
					"profit_gap",
					"final_market_share",
					"avg_market_share",
					"avg_agent_price",
					"price_volatility",
				],
				"value": [
					evaluation_scenario,
					total_steps,
					total_agent_profit,
					total_opponent_profit,
					profit_gap,
					final_market_share,
					avg_market_share,
					avg_agent_price,
					price_volatility,
				],
			}
		)
		st.dataframe(summary_df, use_container_width=True)


st.markdown(
	"""
<div style="position: fixed; bottom: 0; left: 0; right: 0; text-align: center; padding: 14px; 
			background: rgba(255,255,255,0.9); backdrop-filter: blur(20px);
			color: #8E8E93; font-size: 12px; border-top: 1px solid #E5E5EA;">
	RL Evaluation Dashboard • Interactive visualization • Scenario analytics
</div>
""",
	unsafe_allow_html=True,
)
