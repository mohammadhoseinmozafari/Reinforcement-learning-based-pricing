import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
from utils import load_data, PathResolver,moving_average, apply_styles, colors


apply_styles()
window = 10

# ------------------------------------------------------------
# Data loading (unchanged logic)
# ------------------------------------------------------------
st.sidebar.markdown("### Settings")
stage= "train"
training_phase = st.sidebar.radio(
    "Training Phase",
    ["Uniform Pricing"], #"Behavior Based Pricing"],
    index=0
)

opp_type = st.sidebar.radio(
       "Opponent Type",
        ["BBP Opponent" , "Uniform Pricing Opponent"]
    )

run = 1
path_resolver = PathResolver()
path = "experiments/pricing/train/runs/1"

data = load_data(f"{path}/metrics_final.json")

# The general pricing environment emits all three agent and opponent prices.
df_metrics = pd.DataFrame({
    'episode': range(len(data['episode_rewards'])),
    'rewards': data['episode_rewards'],
    'profits': data['episode_profits'],
    'opponent_profits': data['episode_opponent_profits'],
    'uniform_price': data['episode_uniform_prices'],
    'new_price': data['episode_new_prices'],
    'old_price': data['episode_old_prices'],
    'opponent_uniform_price': data['episode_opponent_prices_uniform'],
    'opponent_new_price': data['episode_opponent_prices_new'],
    'opponent_old_price': data['episode_opponent_prices_old'],
    'market_share': data['episode_market_shares'],
})
# Derived metrics
final_avg_reward = np.mean(data['episode_rewards'][-50:]) if len(data['episode_rewards']) >= 50 else np.mean(data['episode_rewards'])
final_market_share = data['episode_market_shares'][-1]
best_reward = max(data['episode_rewards'])
total_episodes = len(data['episode_rewards'])
# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Visualization Controls")
smoothing = st.sidebar.slider("Smoothing window", 1, 30, 10)
window = smoothing
if st.sidebar.checkbox("Show raw data", value=False):
    st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📋 Raw Training Data")
    data_option = st.selectbox(
        "Select data to view",
        ["Episode Metrics", "Losses", "Alphas", "Evaluation Rewards"]
    )
    if data_option == "Episode Metrics":
        st.dataframe(df_metrics, use_container_width=True)
    elif data_option == "Losses":
        st.dataframe(pd.DataFrame({
            'Step': range(len(data['critic_losses'])),
            'Critic Loss': data['critic_losses'],
            'Actor Loss': data['actor_losses']
        }), use_container_width=True)
    elif data_option == "Alphas":
        st.dataframe(pd.DataFrame({
            'Step': range(len(data['alphas'])),
            'Alpha': data['alphas']
        }), use_container_width=True)
    else:
        st.dataframe(pd.DataFrame({
            'Episode': range(len(data['eval_rewards'])),
            'Reward': data['eval_rewards']
        }), use_container_width=True)

st.sidebar.markdown("### 💾 Export Data")
if st.sidebar.button("Export metrics as CSV"):
    csv = df_metrics.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="rl_training_metrics.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown('<div class="hero-title">Training Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Reinforcement Learning Metrics</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Metric cards – four columns with subtle accents
# ------------------------------------------------------------
col0, col1, col2, col3, col4  = st.columns(5)

with col0:
    st.markdown(f"""
    <div class="metric-card">
            <div class="metric-label" style="color:#{colors['harsh_purple'][1:]};">Opponent Type</div>
        <div class="metric-value" style=font-size:20px >{opp_type}</div>
        
    </div>
    """, unsafe_allow_html=True)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label" style="color:#{colors['soft_blue'][1:]};">Total Episodes</div>
        <div class="metric-value">{total_episodes}</div>
        <div class="metric-trend">Training progress</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    start_reward = np.mean(data['episode_rewards'][:50])
    delta_pct = ((final_avg_reward - start_reward) / abs(start_reward) * 100) if start_reward != 0 else 0
    trend_class = "negative" if delta_pct < 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label" style="color:#{colors['soft_green'][1:]};">Avg Reward (last 50)</div>
        <div class="metric-value">{final_avg_reward:.1f}</div>
        <div class="metric-trend {trend_class}">{delta_pct:+.1f}% since start</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label" style="color:#{colors['soft_orange'][1:]};">Best Reward</div>
        <div class="metric-value">{best_reward:.1f}</div>
        <div class="metric-trend">⭐ Peak performance</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    market_leader = final_market_share > 50
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label" style="color:#{colors['soft_pink'][1:]};">Market Share</div>
        <div class="metric-value">{final_market_share:.1f}%</div>
        <div class="metric-trend" style="color:{colors['soft_green'] if market_leader else colors['soft_red']};">
            {'📈 Dominant' if market_leader else '📉 Behind'}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="apple-divider"></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Tabs – Organized viewing
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Training Progress", "💰 Economics", "🧠 Learning Dynamics", "📊 Evaluation"])

# ---------- Tab 1: Training Progress ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        ma_rewards = moving_average(data['episode_rewards'], window= window)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_metrics['episode'],
                y= data['episode_rewards'],
                name = "Episode Rewards",
                mode= 'lines',
                line=dict(
                    color=colors['harsh_blue'],
                    width=3,
                
                ),
                opacity=0.7

            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([df_metrics['episode'], df_metrics['episode'][::-1]]),
                y=np.concatenate([data['episode_rewards'], [0.0]*len(df_metrics)]),
                name = "Area",
                showlegend=False,
                fillcolor=colors['soft_blue'],
                line=dict(
                    width=0
                ),
                fill='toself',
                opacity=0.2

            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_metrics['episode'],
                y= np.concatenate([np.repeat(np.nan, 9), ma_rewards]),
                name = f"Moving Avg ({window})",
                mode= 'lines',
                line=dict(
                    color=colors['soft_red'],
                    width=3
                )
            )
        )
        fig.add_hline(y=0.0, line_dash="dash", opacity=0.7, line_color='black')
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#E5E5EA',
                zeroline=False,
                tickfont=dict(size=12)
            ),
            hovermode='x unified',

        )

        st.markdown("#### Episode Rewards")
        st.plotly_chart(fig)
        

    with col2:
        
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
                x=df_metrics['episode'],
                y=df_metrics['market_share'],
                name='Market Share',
                mode='lines',
                line=dict(
                    color=colors['harsh_teal'],
                    width=3
                )
                ,
                opacity=1.0
            )
        )
        fig.add_trace(go.Scatter(
                x=df_metrics['episode'],
                y=df_metrics['market_share'],
                fill='tonexty',
                fillcolor='rgba(0, 255, 0, 0.2)',  # Green with opacity
                line=dict(width=0),
                showlegend=False,
                name='Above 0.5'
            ))
        df_metrics['above_05'] = df_metrics['market_share'].apply(lambda x: max(x, 0.5))
        df_metrics['below_05'] = df_metrics['market_share'].apply(lambda x: min(x, 0.5))

        fig.add_trace(go.Scatter(
                x=pd.concat([df_metrics['episode'], df_metrics['episode'][::-1]]),
                y=pd.concat([df_metrics['below_05'], pd.Series([0.5]*len(df_metrics))]),
                fill='toself',
                fillcolor=colors['soft_red'],  # Green
                line=dict(width=0),
                showlegend=False,
                name='Below 0.5',
                opacity=0.3
            ))
        fig.add_trace(go.Scatter(
            x=pd.concat([df_metrics['episode'], df_metrics['episode'][::-1]]),
            y=pd.concat([df_metrics['above_05'], pd.Series([0.5]*len(df_metrics))]),
            fill='toself',
            fillcolor=colors['soft_teal'],  # Green
            line=dict(width=0),
            showlegend=False,
            name='Above 0.5',
            opacity=0.3
        ))
        fig.add_hline(y=0.5, line_dash="dash", opacity = 0.7, line_color = 'black')
        # Apple-inspired layout
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#E5E5EA',
                zeroline=False,
                tickfont=dict(size=12)
            ),
            hovermode='x unified',

        )

        st.markdown("#### Episode Market Shares")
        st.plotly_chart(fig)
        

    with col1:
        st.markdown("#### Profit Dynamics")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_metrics['episode'],
            y=data['episode_profits'],
            name='Agent Profit',
            mode='lines',
            line=dict(color=colors['harsh_teal'], width=2),
        
        ))
        fig.add_trace(go.Scatter(
            x=df_metrics['episode'],
            y=data['episode_opponent_profits'],
            name='Opponent Profit',
            mode='lines',
            line=dict(color=colors['harsh_pink'], width=2),
        

        ))
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#E5E5EA',
                zeroline=False,
                tickfont=dict(size=12)
            ),
            hovermode='x unified',

        )
        st.plotly_chart(fig)
    
    with col2:
        st.markdown("#### Price Competition")
        fig = go.Figure()
        price_traces = (
            ('episode_uniform_prices', 'Agent Uniform', colors['harsh_teal'], 'solid'),
            ('episode_new_prices', 'Agent BBP New', colors['harsh_blue'], 'solid'),
            ('episode_old_prices', 'Agent BBP Old', colors['harsh_purple'], 'solid'),
            ('episode_opponent_prices_uniform', 'Opponent Uniform', colors['harsh_pink'], 'dot'),
            ('episode_opponent_prices_new', 'Opponent BBP New', colors['harsh_blue'], 'dot'),
            ('episode_opponent_prices_old', 'Opponent BBP Old', colors['harsh_purple'], 'dot'),
        )
        for key, name, color, dash in price_traces:
            fig.add_trace(go.Scatter(
                x=df_metrics['episode'],
                y=data[key],
                name=name,
                mode='lines',
                line=dict(color=color, width=2, dash=dash),
            ))
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='#E5E5EA', zeroline=False),
            hovermode='x unified',
        )
        st.plotly_chart(fig)

        


# ---------- Tab 2: Economics ----------
with tab2:
    st.markdown("#### Profit Comparison (Agent vs Opponent)")
    cum_agent = np.cumsum(data['episode_profits'])
    cum_opponent = np.cumsum(data['episode_opponent_profits'])
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
                x=df_metrics['episode'],
                y=cum_agent,
                name='Cumulative Agent Profit',
                mode='lines',
                line=dict(
                    color=colors['harsh_teal'],
                    width=3
                )
                ,
                opacity=1.0
            )
        )
    fig.add_trace(go.Scatter(
                x=df_metrics['episode'],
                y=cum_opponent,
                name='Cumulative Opponent Profit',
                mode='lines',
                line=dict(
                    color=colors['harsh_pink'],
                    width=3
                )
                ,
                fill='tonexty',
                fillcolor = 'rgba(189, 238, 226, 0.3)',
                opacity=1.0
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
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#E5E5EA',
                zeroline=False,
                tickfont=dict(size=12)
            ),
            hovermode='x unified',

        )
    st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Profit Distribution")
        st.metric(
            "Total Agent Profit",
            f"{cum_agent[-1]:.1f}",
            delta=f"{((cum_agent[-1] - cum_opponent[-1]) / abs(cum_opponent[-1]) * 100):.1f}% vs opponent"
        )
    with col2:
        st.markdown("#### Final Agent Prices")
        price_cols = st.columns(3)
        price_cols[0].metric("Uniform", f"{data['episode_uniform_prices'][-1]:.2f}")
        price_cols[1].metric("BBP New", f"{data['episode_new_prices'][-1]:.2f}")
        price_cols[2].metric("BBP Old", f"{data['episode_old_prices'][-1]:.2f}")
# ---------- Tab 3: Learning Dynamics ----------
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
                    x=df_metrics['episode'],
                    y=data['critic_losses'],
                    name='Critic Losses',
                    mode='lines',
                    line=dict(
                        color=colors['harsh_purple'],
                        width=3
                    )
                    ,
                    opacity=0.7
                )
            )
        fig.add_trace(go.Scatter(
                    x=df_metrics['episode'],
                    y=data['actor_losses'],
                    name='Actor Losses',
                    mode='lines',
                    line=dict(
                        color=colors['harsh_blue'],
                        width=3
                    ),
                    opacity=0.7
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
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=12)
                ),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#E5E5EA',
                    zeroline=False,
                    tickfont=dict(size=12)
                ),
                hovermode='x unified',

            )
        st.markdown("#### Critic & Actor Losses")

        st.plotly_chart(fig)
        
    with col2:
        st.markdown("#### Alpha (Entropy Coefficient)")
        fig= go.Figure()
        fig.add_trace(go.Scatter(
                x=df_metrics['episode'],
                y=data['alphas'],
                name='Alpha',
                mode='lines',
                line=dict(
                    color=colors['soft_red'],
                    width=3
                ),
                opacity=0.7,
            
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([df_metrics['episode'], df_metrics['episode'][::-1]]),
                y=np.concatenate([data['alphas'], [0.0]*len(df_metrics)]),
                name = "Area",
                showlegend=False,
                fillcolor=colors['soft_red'],
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
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=12)
                ),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#E5E5EA',
                    zeroline=False,
                    tickfont=dict(size=12)
                ),
                hovermode='x unified',

            )

        st.plotly_chart(fig)
            
        st.caption("Controls exploration-exploitation balance")

# ---------- Tab 4: Evaluation ----------
with tab4:
    st.markdown("#### Evaluation Rewards")
    eval_df = pd.DataFrame({
        'Evaluation Episode': range(len(data['eval_rewards'])),
        'Reward': data['eval_rewards']
    })
    st.bar_chart(eval_df.set_index('Evaluation Episode'), color=colors['soft_teal'], height=400)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Eval Reward", f"{np.mean(data['eval_rewards']):.1f}")
    with col2:
        st.metric("Best Eval Reward", f"{max(data['eval_rewards']):.1f}")
    with col3:
        st.metric("Eval Episodes", len(data['eval_rewards']))


# ------------------------------------------------------------
# Footer – subtle, anchored
# ------------------------------------------------------------
st.markdown("""
<div style="position: fixed; bottom: 0; left: 0; right: 0; text-align: center; padding: 14px; 
            background: rgba(255,255,255,0.9); backdrop-filter: blur(20px);
            color: #8E8E93; font-size: 12px; border-top: 1px solid #E5E5EA;">
    RL Training Dashboard • Interactive visualization • Real‑time updates
</div>
""", unsafe_allow_html=True)
