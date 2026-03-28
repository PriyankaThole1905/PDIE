import streamlit as st
import time
import auth_db

# Ensure DB is initialized on every import
auth_db.init_db()


def show_login_page():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stHeader"] { display: none !important; }
        footer { display: none !important; }
        
        html, body, .stApp, [data-testid="stAppViewContainer"], .main {
            font-family: 'Inter', sans-serif !important;
            background-color: #0A1628 !important;
            background-image: none !important;
        }
        
        /* RIGHT SIDE SPLIT GRADIENT */
        .stApp::before, [data-testid="stAppViewContainer"]::before {
            content: '';
            position: fixed;
            top: 0;
            right: 0;
            width: 55%;
            height: 100vh;
            background: linear-gradient(135deg, #00539B 0%, #003d73 100%) !important;
            z-index: 0;
            pointer-events: none;
        }
        
        div[data-testid="stHorizontalBlock"] {
            align-items: center !important;
            min-height: 100vh !important;
        }

        div[data-testid="column"]:nth-of-type(1) {
            padding: 0 5% !important;
            position: relative;
            z-index: 10;
        }
        div[data-testid="column"]:nth-of-type(2) {
            padding: 0 5% !important;
            position: relative;
            z-index: 10;
        }

        div[data-testid="stTextInput"] input {
            background-color: rgba(255,255,255,0.05) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 10px !important;
            font-size: 1rem !important;
            padding: 0.75rem 1rem !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #00A3E0 !important;
            box-shadow: 0 0 0 2px rgba(0,163,224,0.25) !important;
        }
        div[data-testid="stTextInput"] label p {
            color: #94a3b8 !important;
            font-size: 0.72rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            font-weight: 700 !important;
        }

        [data-testid="stButton"] button {
            width: 100% !important;
            background: linear-gradient(135deg, #00539B 0%, #00A3E0 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 1.4rem !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            margin-top: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 83, 155, 0.4) !important;
        }
        [data-testid="stButton"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 83, 155, 0.6) !important;
            background: linear-gradient(135deg, #00A3E0 0%, #00d2ff 100%) !important;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .metric-card {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 1.8rem;
            border-radius: 16px;
            backdrop-filter: blur(12px);
            width: 240px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            animation: fadeUp 0.8s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
            opacity: 0;
            transition: transform 0.3s ease;
        }
        .metric-card:hover { transform: translateY(-5px); border-color: rgba(255,255,255,0.4); }
        .m1 { animation-delay: 0.2s; }
        .m2 { animation-delay: 0.4s; }

        /* Alert overrides for dark background */
        [data-testid="stAlert"] {
            background: rgba(220,38,38,0.15) !important;
            border: 1px solid rgba(220,38,38,0.3) !important;
            border-radius: 10px !important;
        }
        [data-testid="stAlert"] p { color: #fca5a5 !important; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([4.5, 5.5])

    with c1:
        st.markdown(
            '<div style="margin-bottom: 3rem;">'
            '<h2 style="color:#ffffff; font-weight:900; letter-spacing:1px; margin:0; font-size:1.8rem; display:flex; align-items:center; gap:10px;">'
            '<span style="background:#00539B; width:32px; height:32px; display:inline-block; border-radius:6px; position:relative;">'
            '<span style="position:absolute; width:16px; height:3px; background:#fff; top:10px; left:8px; border-radius:2px;"></span>'
            '<span style="position:absolute; width:16px; height:3px; background:#fff; top:18px; left:8px; border-radius:2px;"></span>'
            "</span>"
            'BARCLAYS<span style="color:#00A3E0;">.</span></h2>'
            '<div style="color:#00A3E0; font-size:0.8rem; font-weight:700; letter-spacing:1px; margin-top:0.5rem; text-transform:uppercase;">Collections Manager Portal</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<h1 style="color:#ffffff; font-size:2.4rem; font-weight:800; margin:0 0 0.3rem 0; letter-spacing:-0.5px;">Welcome Back</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.95rem; margin:0 0 2.5rem 0; font-weight:500;">Sign in to monitor your portfolio.</p>',
            unsafe_allow_html=True,
        )

        email = st.text_input(
            "📧 Email Address", key="login_email", placeholder="you@barclays.com"
        )
        password = st.text_input(
            "🔒 Password", type="password", key="login_pass", placeholder="••••••••"
        )

        st.markdown(
            """
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem; margin-bottom:0.5rem;">
                <div style="color:#94a3b8; font-size:0.85rem;"><input type="checkbox" style="margin-right:8px;" checked> Remember me</div>
                <a href='#' style='color:#00A3E0; font-size:0.85rem; font-weight:600; text-decoration:none;'>Forgot password?</a>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("🔐 Sign In →"):
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                with st.spinner("Authenticating…"):
                    result = auth_db.authenticate_user(email, password)
                    time.sleep(0.8)  # UX delay

                if result["ok"]:
                    st.session_state["app_state"] = "dashboard"
                    st.session_state["logged_in"] = True
                    st.session_state["user_id"] = result["id"]
                    st.session_state["full_name"] = result["full_name"]
                    st.session_state["email"] = result["email"]
                    st.session_state["role"] = result["role"]
                    st.session_state["username"] = result["email"]
                    st.session_state["analyst_name"] = result["full_name"]
                    st.session_state["assigned_role"] = result["role"]
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")

        st.markdown(
            """
            <p style='color:#64748b; font-size:0.85rem; text-align:center; margin-top:1.5rem;'>
            Don't have an account? 
            </p>
        """,
            unsafe_allow_html=True,
        )

        if st.button("✨ Create New Account"):
            st.session_state["app_state"] = "signup"
            st.rerun()

    with c2:
        st.markdown(
            '<div style="display:flex; flex-direction:column; justify-content:center; align-items: flex-start; height:100%; position:relative; z-index:5;">'
            '<div style="background: rgba(255,255,255,0.1); padding: 0.4rem 1rem; border-radius:20px; color:#e0f2fe; font-size:0.8rem; font-weight:700; letter-spacing:1px; margin-bottom:1.5rem; border: 1px solid rgba(255,255,255,0.2);">PDIE • RISK INTELLIGENCE PLATFORM</div>'
            '<h1 style="color:#ffffff; font-size:4.5rem; font-weight:900; line-height:1.05; margin-bottom:1.5rem; letter-spacing:-1.5px; text-shadow: 0 4px 15px rgba(0,0,0,0.2);">'
            "AI-Powered Risk<br>Intelligence."
            "</h1>"
            '<p style="color:#e0f2fe; font-size:1.3rem; font-weight:400; max-width:85%; line-height:1.6; margin-bottom:4rem; opacity:0.95; border-left: 4px solid #4ade80; padding-left: 1rem;">'
            "Predict delinquency 2–4 weeks ahead with <strong>86.3% accuracy</strong>. Streamline interventions and minimize aggregate portfolio default."
            "</p>"
            '<div style="display:flex; gap:2rem;">'
            '<div class="metric-card m1">'
            '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7dd3fc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:0.8rem;"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>'
            '<div style="color:#ffffff; font-size:2.5rem; font-weight:900; letter-spacing:-1px;">10k+</div>'
            '<div style="color:#94a3b8; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-top:0.2rem;">Accounts Monitored</div>'
            "</div>"
            '<div class="metric-card m2">'
            '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:0.8rem;"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>'
            '<div style="color:#ffffff; font-size:2.5rem; font-weight:900; letter-spacing:-1px;">86.3%</div>'
            '<div style="color:#94a3b8; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-top:0.2rem;">Model AUC</div>'
            "</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
