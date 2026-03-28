import streamlit as st
import base64
import os

def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

def show_landing_page():
    img_path = os.path.join(os.path.dirname(__file__), "hero_image2.png")
    bg_b64 = get_base64_of_bin_file(img_path)
    if bg_b64:
        bg_url = f"data:image/png;base64,{bg_b64}"
    else:
        bg_url = "https://images.unsplash.com/photo-1573164713988-8665fc963095?ixlib=rb-4.0.3&auto=format&fit=crop&w=1469&q=80"

    css = """
        <style>
        /* Hide default Streamlit elements */
        [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stHeader"] { display: none !important; }
        footer { display: none !important; }
        
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
            margin: 0 !important;
            background-color: #ffffff !important;
        }

        /* Top Nav Styling */
        .nav-container {
            padding: 1rem 5%;
            background: #ffffff;
            border-bottom: 1px solid #f1f5f9;
        }

        /* Hero Styling */
        .hero-banner {
            background-color: #00A3E0;
            padding: 10px;
            margin: 2.5rem 5%;
            display: flex;
            min-height: 400px;
        }
        .hero-left {
            background-color: #001f5c;
            padding: 4rem;
            width: 45%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .hero-right {
            width: 55%;
            background: url("HERO_IMG_URL") center/cover no-repeat;
        }

        /* Feature Cards */
        .features-wrapper {
            padding: 3rem 5% 5rem 5%;
            text-align: center;
        }
        .feature-grid {
            display: flex;
            gap: 2rem;
            margin-top: 3rem;
            justify-content: center;
        }
        .f-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 2.5rem;
            flex: 1;
            text-align: left;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            transition: transform 0.2s ease;
        }
        .f-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 15px rgba(0,0,0,0.05);
        }

        /* Button Overrides */
        .stButton > button {
            background-color: #00539B !important;
            color: white !important;
            border: none !important;
            border-radius: 24px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 700 !important;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            background-color: #00A3E0 !important;
            color: white !important;
        }
        </style>
    """
    st.markdown(css.replace("HERO_IMG_URL", bg_url), unsafe_allow_html=True)

    # Navigation Bar
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 5, 1])
    with col1:
        st.markdown(
            '<div style="color:#00A3E0; font-size:1.8rem; font-weight:900; letter-spacing:1px; margin-top:0.2rem;">'
            '<span style="color:#00A3E0; margin-right:8px;">🦅</span> BARCLAYS'
            '</div>', 
            unsafe_allow_html=True
        )
    with col3:
        if st.button("Log in"):
            st.session_state['app_state'] = 'login'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Hero Banner
    hero_html = (
        '<div class="hero-banner">'
        '<div class="hero-left">'
        '<h1 style="color:#ffffff; font-size:3.5rem; font-weight:900; margin:0 0 1rem 0; line-height:1.1; letter-spacing:-1px;">Empower your portfolio.<br>Bank on AI.</h1>'
        '<p style="color:#e0f2fe; font-size:1.1rem; margin:0 0 2rem 0; font-weight:500;">Boost your recovery rates with real-time agentic intelligence from our PDIE team.</p>'
        '<div><span style="display:inline-block; background:#ffffff; color:#001f5c; padding:0.8rem 1.8rem; border-radius:30px; font-weight:800; font-size:1rem;">Access analyst portal &rarr;</span></div>'
        '</div>'
        '<div class="hero-right"></div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    # Features Section
    features_html = (
        '<div class="features-wrapper">'
        '<h2 style="color:#0f172a; font-size:2rem; font-weight:800; margin-bottom:0.8rem;">Supporting your portfolio, every step of the way</h2>'
        '<p style="color:#64748b; font-size:1.1rem; margin-bottom:0;">We\'re here to help you identify high-risk assets early. Our AI engines drive superior performance.</p>'
        '<div class="feature-grid">'
        '<div class="f-card">'
        '<div style="font-size:2.5rem; margin-bottom:1rem;">🤖</div>'
        '<h3 style="color:#0f172a; font-size:1.3rem; font-weight:800; margin:0 0 0.8rem 0;">24/7 Agentic Support</h3>'
        '<p style="color:#475569; font-size:0.95rem; line-height:1.6; margin:0;">Send custom outreach messages automatically. Switch to live agents instantly when recovery probability drops.</p>'
        '</div>'
        '<div class="f-card">'
        '<div style="font-size:2.5rem; margin-bottom:1rem;">📊</div>'
        '<h3 style="color:#0f172a; font-size:1.3rem; font-weight:800; margin:0 0 0.8rem 0;">10,000+ Profiles Scored</h3>'
        '<p style="color:#475569; font-size:0.95rem; line-height:1.6; margin:0;">Join an ecosystem utilizing high-accuracy XGBoost models to calculate real-time PD and identify distress early.</p>'
        '</div>'
        '<div class="f-card">'
        '<div style="font-size:2.5rem; margin-bottom:1rem;">⚡</div>'
        '<h3 style="color:#0f172a; font-size:1.3rem; font-weight:800; margin:0 0 0.8rem 0;">Automated NPV Recovery</h3>'
        '<p style="color:#475569; font-size:0.95rem; line-height:1.6; margin:0;">Benefit from Monte Carlo simulations that identify the exact recovery pathway needed to maximize NPV returns.</p>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(features_html, unsafe_allow_html=True)

