"""
PDIE Design System - Centralized Theme & Styling
Unified design tokens for consistent professional UI across all pages.

Author: PDIE Team | Barclays Hack-O-Hire 2026
"""

import streamlit as st

# ═════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE - Barclays × Modern Fintech
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {
    # Primary
    "primary": "#00539B",
    "primary_dark": "#003d73",
    "primary_light": "#00A3E0",
    # Backgrounds
    "bg_dark": "#0A1628",
    "bg_card": "#1E293B",
    "bg_surface": "#0f172a",
    # Status
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6",
    # Text
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    # Risk Tiers
    "risk_critical": "#EF4444",
    "risk_high": "#F97316",
    "risk_medium": "#FACC15",
    "risk_low": "#22C55E",
    # Gradients
    "gradient_header": "linear-gradient(135deg, #0f172a 0%, #00395D 50%, #00A3E0 100%)",
    "gradient_button": "linear-gradient(135deg, #00539B 0%, #00A3E0 100%)",
    "gradient_card": "linear-gradient(180deg, rgba(15,23,42,0.95) 0%, rgba(0,57,93,0.98) 100%)",
}


# ═════════════════════════════════════════════════════════════════════════════
# TYPOGRAPHY
# ═════════════════════════════════════════════════════════════════════════════

FONTS = {
    "family": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    "heading_weight": 800,
    "body_weight": 400,
}


# ═════════════════════════════════════════════════════════════════════════════
# SPACING & SIZING
# ═════════════════════════════════════════════════════════════════════════════

SIZING = {
    "border_radius_sm": "8px",
    "border_radius_md": "12px",
    "border_radius_lg": "16px",
    "border_radius_xl": "20px",
    "border_radius_full": "9999px",
    "spacing_xs": "0.25rem",
    "spacing_sm": "0.5rem",
    "spacing_md": "1rem",
    "spacing_lg": "1.5rem",
    "spacing_xl": "2rem",
    "sidebar_width": "280px",
}


# ═════════════════════════════════════════════════════════════════════════════
# CENTRALIZED CSS - Complete Design System
# ═════════════════════════════════════════════════════════════════════════════


def get_theme_css():
    """Returns complete theme CSS for injection into Streamlit pages."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* === GLOBAL RESET === */
html, body, [class*="st-"], .stMarkdown, .stText {{
    font-family: {FONTS["family"]} !important;
    color: #1e293b; /* Premium dark slate for light theme readability */
}}


/* Hide default Streamlit elements */
[data-testid="collapsedControl"] {{ display: none !important; }}
[data-testid="stHeader"] {{ display: none !important; }}
footer {{ display: none !important; }}

/* === PREMIUM SIDEBAR === */
section[data-testid="stSidebar"] {{
    background: {COLORS["gradient_card"]} !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}}
section[data-testid="stSidebar"] * {{
    color: {COLORS["text_primary"]} !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.1) !important;
}}

/* === HEADER BANNER === */
.main-header {{
    background: {COLORS["gradient_header"]};
    padding: 2.5rem 3rem;
    border-radius: {SIZING["border_radius_lg"]};
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 12px 40px rgba(0, 57, 93, 0.4);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.15);
}}
.main-header::before {{
    content: '';
    position: absolute;
    top: -60%;
    right: -15%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
    animation: slowPulse 8s infinite alternate;
}}
.main-header h1 {{ 
    margin: 0 0 0.4rem 0; 
    font-weight: {FONTS["heading_weight"]}; 
    font-size: 2.2rem; 
    letter-spacing: -0.8px; 
    color: white !important; /* Force white for visibility on dark banner */
}}
.main-header p {{ 
    margin: 0; 
    opacity: 0.9; 
    font-size: 1.05rem; 
    color: white !important; /* Force white for visibility on dark banner */
}}


@keyframes slowPulse {{
    0% {{ transform: scale(1); opacity: 0.6; }}
    100% {{ transform: scale(1.1); opacity: 1; }}
}}

/* === KPI CARDS === */
.kpi-card {{
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 163, 224, 0.2);
    border-radius: {SIZING["border_radius_lg"]};
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}}
.kpi-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0, 163, 224, 0.15);
    border-color: rgba(0, 163, 224, 0.5);
}}
.kpi-card .kpi-icon {{ font-size: 1.8rem; margin-bottom: 0.4rem; }}
.kpi-card .kpi-label {{ font-size: 0.8rem; color: {COLORS["text_muted"]}; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700; }}
.kpi-card .kpi-value {{ font-size: 2rem; font-weight: 900; color: #0f172a; margin: 0.2rem 0; }}
.kpi-card .kpi-delta {{ font-size: 0.85rem; font-weight: 600; display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 12px; background: #f1f5f9; }}

/* === RISK BADGES === */
.risk-badge {{
    padding: 0.5rem 1.2rem;
    border-radius: 30px;
    font-weight: 800;
    font-size: 0.85rem;
    display: inline-block;
}}
.risk-critical {{
    background: linear-gradient(135deg, #ef4444, #991b1b);
    color: white;
    box-shadow: 0 4px 12px rgba(220,38,38,0.4);
    animation: pulseRed 2s infinite;
}}
.risk-high {{
    background: linear-gradient(135deg, #f97316, #c2410c);
    color: white;
    box-shadow: 0 4px 12px rgba(249,115,22,0.3);
}}
.risk-medium {{
    background: linear-gradient(135deg, #facc15, #ca8a04);
    color: #1e293b;
}}
.risk-low {{
    background: linear-gradient(135deg, #22c55e, #15803d);
    color: white;
}}

@keyframes pulseRed {{
    0% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.6); }}
    70% {{ box-shadow: 0 0 0 8px rgba(220, 38, 38, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }}
}}

/* === PREMIUM BUTTONS === */
.stButton>button {{
    background: {COLORS["gradient_button"]};
    color: white;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: {SIZING["border_radius_md"]};
    padding: 0.7rem 2.2rem;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.4);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    font-size: 0.95rem;
}}
.stButton>button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 163, 224, 0.5);
    border-color: {COLORS["primary_light"]};
}}

/* === INPUTS === */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stTextArea>div>div>textarea {{
    background: #f8fafc !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: {SIZING["border_radius_md"]} !important;
    color: #1e293b !important; /* Dark text for visibility on light backgrounds */
    font-weight: 500 !important;
}}
.stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {{
    border-color: {COLORS["primary_light"]} !important;
    box-shadow: 0 0 0 2px rgba(0,163,224,0.15) !important;
    background: white !important;
}}
.stTextInput label, .stNumberInput label, .stTextArea label {{
    color: #475569 !important;
    font-weight: 600 !important;
}}


/* === TABS === */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px !important;
    border-bottom: 2px solid #e2e8f0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: {SIZING["border_radius_md"]} {SIZING["border_radius_md"]} 0 0 !important;
    padding: 0.8rem 1.6rem !important;
    font-weight: 700 !important;
    color: {COLORS["text_muted"]} !important;
    transition: all 0.2s ease !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: #0f172a !important;
    background: rgba(241,245,249,0.8) !important;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: {COLORS["primary"]} !important;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"]::after {{
    content: '' !important;
    position: absolute !important;
    bottom: -2px !important;
    left: 0 !important;
    width: 100% !important;
    height: 3px !important;
    background: {COLORS["gradient_button"]} !important;
}}

/* === CARDS === */
.info-card, .hub-card {{
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: {SIZING["border_radius_lg"]};
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(16px);
}}
.info-card:hover, .hub-card:hover {{
    border-color: rgba(0, 163, 224, 0.3);
}}

/* === METRIC PILL === */
.metric-pill {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: {SIZING["border_radius_md"]};
    padding: 0.8rem 1rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}}
.metric-pill .mp-val {{
    font-size: 1.4rem;
    font-weight: 800;
}}
.metric-pill .mp-lbl {{
    font-size: 0.7rem;
    color: {COLORS["text_secondary"]};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.2rem;
}}

/* === SIDEBAR BADGE === */
.sidebar-badge {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: {SIZING["border_radius_md"]};
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.sidebar-badge:hover {{ background: rgba(255,255,255,0.1); }}
.sidebar-badge .badge-val {{
    font-weight: 800;
    color: {COLORS["primary_light"]} !important;
}}

/* === NAVIGATION === */
.nav-section {{
    margin-bottom: 1rem;
}}
.nav-section-title {{
    font-size: 0.7rem;
    color: {COLORS["text_muted"]};
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 0.5rem 0.75rem;
    font-weight: 700;
}}
.nav-item {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    border-radius: {SIZING["border_radius_md"]};
    color: {COLORS["text_secondary"]};
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    margin: 0.2rem 0;
}}
.nav-item:hover {{
    background: rgba(255,255,255,0.05);
    color: {COLORS["text_primary"]};
}}
.nav-item.active {{
    background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["primary_light"]});
    color: white;
    box-shadow: 0 4px 12px rgba(0, 163, 224, 0.3);
}}
.nav-item .nav-icon {{
    font-size: 1.2rem;
}}

/* === NUCLEAR OPTION SIDEBAR BUTTON FIX (REMOVE WHITE BOXES) === */
section[data-testid="stSidebar"] .stButton {{
    background: transparent !important;
    margin: 0 !important;
    padding: 0 !important;
}}

section[data-testid="stSidebar"] .stButton button {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: {COLORS["text_secondary"]} !important;
    width: 100% !important;
    padding: 0.8rem 1rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
    border-radius: {SIZING["border_radius_md"]} !important;
    transition: all 0.2s ease !important;
    margin: 0.2rem 0 !important;
    min-height: 48px !important;
}}

section[data-testid="stSidebar"] .stButton button:hover {{
    background: rgba(255,255,255,0.08) !important;
    color: {COLORS["text_primary"]} !important;
}}

/* Active State Branding - Force the gradient correctly */
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["primary_light"]}) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(0, 83, 155, 0.3) !important;
    border: none !important;
}}

/* Ensure text inside button is visible */
section[data-testid="stSidebar"] .stButton button p,
section[data-testid="stSidebar"] .stButton button div,
section[data-testid="stSidebar"] .stButton button span {{
    color: inherit !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}}

/* Ensure the container doesn't add any background */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div > div {{
    background: transparent !important;
}}





/* === STATUS INDICATORS === */
.status-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}
.status-dot.online {{ background: {COLORS["success"]}; box-shadow: 0 0 8px {COLORS["success"]}; }}
.status-dot.offline {{ background: {COLORS["danger"]}; }}
.status-dot.warning {{ background: {COLORS["warning"]}; }}

/* === LOADING STATES & ANIMATIONS === */
@keyframes shimmer {{
    0% {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}

.skeleton-loader {{
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: {SIZING["border_radius_md"]};
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.fade-in {{
    animation: fadeIn 0.3s ease-out forwards;
}}

@keyframes slideIn {{
    from {{ opacity: 0; transform: translateX(-20px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}

.slide-in {{
    animation: slideIn 0.4s ease-out forwards;
}}

/* === TOAST NOTIFICATIONS === */
.toast {{
    position: fixed;
    bottom: 20px;
    right: 20px;
    padding: 1rem 1.5rem;
    border-radius: {SIZING["border_radius_md"]};
    color: white;
    font-weight: 600;
    z-index: 9999;
    animation: slideIn 0.3s ease-out;
}}
.toast.success {{ background: linear-gradient(135deg, {COLORS["success"]}, #16a34a); }}
.toast.error {{ background: linear-gradient(135deg, {COLORS["danger"]}, #dc2626); }}
.toast.warning {{ background: linear-gradient(135deg, {COLORS["warning"]}, #d97706); }}
.toast.info {{ background: linear-gradient(135deg, {COLORS["info"]}, #2563eb); }}

/* === PROGRESS BARS === */
.progress-bar {{
    height: 8px;
    background: rgba(0,0,0,0.1);
    border-radius: 4px;
    overflow: hidden;
}}
.progress-bar .progress-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}}

/* === CUSTOM SCROLLBAR === */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}
::-webkit-scrollbar-track {{
    background: rgba(0,0,0,0.1);
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb {{
    background: {COLORS["primary"]};
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {COLORS["primary_light"]};
}}

/* === CARD HOVER EFFECTS === */
.hover-card {{
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}}
.hover-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0, 163, 224, 0.15);
}}

/* === TOOLTIP === */
.tooltip {{
    position: relative;
    display: inline-block;
}}
.tooltip .tooltip-text {{
    visibility: hidden;
    background: {COLORS["bg_dark"]};
    color: {COLORS["text_primary"]};
    text-align: center;
    padding: 0.5rem 1rem;
    border-radius: {SIZING["border_radius_sm"]};
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.8rem;
    white-space: nowrap;
}}
.tooltip:hover .tooltip-text {{
    visibility: visible;
    opacity: 1;
}}

/* === DATA TABLE ENHANCEMENTS === */
.data-table {{
    width: 100%;
    border-collapse: collapse;
}}
.data-table th {{
    background: {COLORS["bg_dark"]};
    color: {COLORS["text_primary"]};
    padding: 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.data-table td {{
    padding: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}}
.data-table tr:hover td {{
    background: rgba(0,163,224,0.05);
}}

</style>
"""


def inject_theme():
    """Inject theme CSS into Streamlit page."""
    st.markdown(get_theme_css(), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════


def get_risk_color(risk_level):
    """Return color based on risk level."""
    if risk_level >= 80:
        return COLORS["risk_critical"]
    elif risk_level >= 65:
        return COLORS["risk_high"]
    elif risk_level >= 50:
        return COLORS["risk_medium"]
    else:
        return COLORS["risk_low"]


def format_currency(amount):
    """Format number as currency."""
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f}Cr"
    elif amount >= 100000:
        return f"₹{amount / 100000:.2f}L"
    elif amount >= 1000:
        return f"₹{amount / 1000:.1f}K"
    return f"₹{amount}"


# ═════════════════════════════════════════════════════════════════════════════
# PAGE HEADER COMPONENT
# ═════════════════════════════════════════════════════════════════════════════


def render_page_header(title, subtitle="", icon=""):
    """Render a unified page header."""
    icon_html = f"<span style='margin-right:0.5rem;'>{icon}</span>" if icon else ""
    st.markdown(
        f"""
    <div class="main-header">
        <h1>{icon_html}{title}</h1>
        <p>{subtitle}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# NAVIGATION COMPONENTS
# ═════════════════════════════════════════════════════════════════════════════


def render_nav_section(title, items, current_page, key_prefix="nav"):
    """Render a collapsible navigation section."""
    st.markdown(f'<div class="nav-section-title">{title}</div>', unsafe_allow_html=True)

    for item in items:
        page_name = item["page"]
        icon = item.get("icon", "")
        label = item.get("label", page_name)

        is_active = current_page == page_name
        active_class = "active" if is_active else ""

        icon_html = f'<span class="nav-icon">{icon}</span>' if icon else ""

        st.markdown(
            f"""
        <div class="nav-item {active_class}" onclick="document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active')); this.classList.add('active');">
            {icon_html}
            <span>{label}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button(label, key=f"{key_prefix}_{page_name}"):
            st.session_state["page"] = page_name
            st.rerun()


def render_sidebar_stats():
    """Render quick stats in sidebar."""
    st.markdown("##### 📊 Quick Stats")
    st.markdown(
        f"""
    <div class="sidebar-badge"><span>Customers</span><span class="badge-val">10,000</span></div>
    <div class="sidebar-badge"><span>Model AUC</span><span class="badge-val">86.3%</span></div>
    <div class="sidebar-badge"><span>Prediction</span><span class="badge-val">21 days</span></div>
    <div class="sidebar-badge"><span>Success Rate</span><span class="badge-val">73.2%</span></div>
    """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# INTERACTIVE COMPONENTS
# ═════════════════════════════════════════════════════════════════════════════


def render_stat_card(title, value, delta="", icon="", color="#00539B"):
    """Render a professional stat card with optional delta."""
    delta_html = ""
    if delta:
        delta_color = "#22c55e" if "+" in str(delta) else "#ef4444"
        delta_html = f'<span style="color:{delta_color}; font-size:0.8rem; margin-left:0.5rem;">{delta}</span>'

    icon_html = (
        f'<span style="font-size:1.5rem; margin-right:0.5rem;">{icon}</span>'
        if icon
        else ""
    )

    st.markdown(
        f"""
    <div class="kpi-card hover-card">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
                <div class="kpi-label">{title}</div>
                <div class="kpi-value" style="color:{color};">{icon_html}{value}{delta_html}</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_risk_badge(risk_level, size="md"):
    """Render a risk badge with appropriate styling."""
    if risk_level >= 80:
        risk_class = "risk-critical"
        label = "CRITICAL"
    elif risk_level >= 65:
        risk_class = "risk-high"
        label = "HIGH"
    elif risk_level >= 50:
        risk_class = "risk-medium"
        label = "MEDIUM"
    else:
        risk_class = "risk-low"
        label = "LOW"

    font_size = "0.75rem" if size == "sm" else "0.85rem"

    st.markdown(
        f"""
    <span class="risk-badge {risk_class}" style="font-size:{font_size};">
        {label} ({risk_level})
    </span>
    """,
        unsafe_allow_html=True,
    )


def render_status_indicator(status, label):
    """Render a status indicator dot with label."""
    if status == "online":
        dot_class = "online"
    elif status == "warning":
        dot_class = "warning"
    else:
        dot_class = "offline"

    st.markdown(
        f"""
    <div style="display:flex; align-items:center; gap:0.5rem;">
        <span class="status-dot {dot_class}"></span>
        <span style="color:#94a3b8; font-size:0.85rem;">{label}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_progress_bar(value, max_value=100, color="#00A3E0", show_label=True):
    """Render a progress bar with percentage."""
    percentage = min(100, max(0, (value / max_value) * 100))

    st.markdown(
        f"""
    <div style="margin:0.5rem 0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:0.25rem;">
            <span style="font-size:0.8rem; color:#64748b;">Progress</span>
            <span style="font-size:0.8rem; font-weight:600; color:{color};">{percentage:.0f}%</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width:{percentage}%; background:{color};"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_toast_message(message, toast_type="info"):
    """Render a toast notification message."""
    icons = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
    icon = icons.get(toast_type, "ℹ️")

    st.markdown(
        f"""
    <div class="toast {toast_type}">
        <span style="margin-right:0.5rem;">{icon}</span>
        {message}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_section_header(title, description="", icon=""):
    """Render a professional section header."""
    icon_html = f'<span style="margin-right:0.5rem;">{icon}</span>' if icon else ""

    desc_html = (
        f'<p style="color:#94a3b8; margin:0.25rem 0 0; font-size:0.85rem;">{description}</p>'
        if description
        else ""
    )

    st.markdown(
        f"""
    <div style="margin: 1.5rem 0 1rem;">
        <h3 style="color:#0f172a; margin:0; font-weight:700; font-size:1.1rem;">
            {icon_html}{title}
        </h3>
        {desc_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_info_row(label, value, icon="", highlight=False):
    """Render an info row with label and value."""
    bg = "rgba(0,163,224,0.05)" if highlight else "transparent"
    border = "1px solid rgba(0,163,224,0.2)" if highlight else "none"

    icon_html = f'<span style="margin-right:0.5rem;">{icon}</span>' if icon else ""

    st.markdown(
        f"""
    <div style="display:flex; justify-content:space-between; padding:0.75rem; background:{bg}; border:{border}; border-radius:8px; margin:0.25rem 0;">
        <span style="color:#64748b; font-weight:600; font-size:0.9rem;">{icon_html}{label}</span>
        <span style="color:#0f172a; font-weight:700; font-size:0.9rem;">{value}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_metric_pill(label, value, color="#00539B", icon=""):
    """Render a metric pill component."""
    icon_html = f'<span style="margin-right:0.25rem;">{icon}</span>' if icon else ""

    st.markdown(
        f"""
    <div class="metric-pill">
        <span class="mp-val" style="color:{color};">{icon_html}{value}</span>
        <span class="mp-lbl">{label}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )
