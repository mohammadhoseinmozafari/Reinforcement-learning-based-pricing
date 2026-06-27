import streamlit as st

def apply_styles():
    """Apply Apple-inspired design system to the app"""
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
        .hero-title {
            font-size: 42px;
            font-weight: 700;
            color: #1D1D1F;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        }
        
        .hero-subtitle {
            font-size: 18px;
            color: #86868B;
            font-weight: 400;
        }

        /*  Cards – floating, airy */
        .dashboard-card {
            background: #FFFFFF;
            border-radius: 18px;
            padding: 40px 32px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.03);
            transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            cursor: pointer;
            text-align: center;
            border: 1px solid #F0F0F5;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 320px;
            position: relative;
            overflow: hidden;
    }
        .dashboard-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08), 0 6px 12px rgba(0, 0, 0, 0.05);
            border-color: #E5E5EA;
    }
        .dashboard-card:active {
            transform: scale(0.98);
    }
                
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
                .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        padding: 20px 0;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Individual card */
    

    
    
    .card-icon {
        font-size: 64px;
        margin-bottom: 16px;
        display: block;
    }
    
    .card-title {
        font-size: 24px;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: -0.3px;
        margin-bottom: 8px;
    }
    
    .card-description {
        font-size: 15px;
        color: #86868B;
        line-height: 1.5;
        max-width: 280px;
        margin: 0 auto;
    }
    
    .card-badge {
        position: absolute;
        top: 16px;
        right: 16px;
        background: #F0F0F5;
        color: #8E8E93;
        font-size: 11px;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 12px;
        letter-spacing: 0.3px;
    }
    
    .card-hover-effect {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(93, 155, 236, 0.05), rgba(93, 155, 236, 0.01));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .dashboard-card:hover .card-hover-effect {
        opacity: 1;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .dashboard-grid {
            grid-template-columns: 1fr;
            gap: 16px;
        }
        .dashboard-card {
            min-height: 200px;
            padding: 24px;
        }
    }
    
    /* Featured section */
    .hero-section {
        text-align: center;
        padding: 40px 0 20px 0;
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
            color: #34C759;
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
    </style>
    """, unsafe_allow_html=True)

def metric_card(label, value, trend=None, trend_value=None):
    """Create an Apple-style metric card"""
    trend_html = ""
    if trend:
        trend_class = "metric-trend" if trend == "positive" else "metric-trend negative"
        trend_html = f'<div class="{trend_class}">{trend_value}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {trend_html}
    </div>
    """