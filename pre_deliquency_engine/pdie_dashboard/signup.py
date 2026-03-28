import streamlit as st
import time
import auth_db

# Ensure DB is initialized on every import
auth_db.init_db()


def show_signup_page():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stHeader"] { display: none !important; }
        footer { display: none !important; }
        
        html, body, .stApp, [data-testid="stAppViewContainer"], .main {
            background-color: #0b1120 !important;
            background-image: none !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        .stApp::before, [data-testid="stAppViewContainer"]::before {
            content: '';
            position: fixed;
            top: 0;
            right: 0;
            width: 58%;
            height: 100vh;
            background: linear-gradient(135deg, #001f5c 0%, #00395D 40%, #00539B 100%) !important;
            z-index: 0;
            pointer-events: none;
        }
        .stApp::after, [data-testid="stAppViewContainer"]::after {
            content:''; position:fixed; top:-20%; right:-10%;
            width: 800px; height: 800px; border-radius: 50%;
            background: radial-gradient(circle, rgba(0,163,224,0.15) 0%, transparent 60%);
            pointer-events: none;
            z-index: 1;
        }
        
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
            margin: 0 !important;
            z-index: 10 !important;
            position: relative;
        }
        
        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            min-height: 100vh !important;
        }
        
        div[data-testid="column"]:first-child {
            width: 42% !important;
            min-width: 42% !important;
            flex: 0 0 42% !important;
            background: transparent !important;
            padding: 3% 5% !important;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        div[data-testid="column"]:last-child {
            width: 58% !important;
            min-width: 58% !important;
            flex: 0 0 58% !important;
            background: transparent !important;
            padding: 0 8% !important;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        div[data-testid="stTextInput"] input {
            background-color: rgba(255,255,255,0.04) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 10px !important;
            padding: 0.75rem 1rem !important;
            font-size: 0.95rem !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #38bdf8 !important;
            background-color: rgba(255,255,255,0.06) !important;
            box-shadow: 0 0 0 2px rgba(56,189,248,0.25) !important;
        }
        div[data-testid="stTextInput"] label p {
            color: #94a3b8 !important;
            font-size: 0.72rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
            font-weight: 700 !important;
            margin-bottom: -0.2rem !important;
        }

        /* Selectbox */
        div[data-testid="stSelectbox"] > div > div {
            background-color: rgba(255,255,255,0.04) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 10px !important;
        }
        div[data-testid="stSelectbox"] label p {
            color: #94a3b8 !important;
            font-size: 0.72rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
            font-weight: 700 !important;
        }

        [data-testid="stButton"] button {
            width: 100% !important;
            background-color: #00A3E0 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 1.3rem !important;
            font-weight: 800 !important;
            font-size: 1rem !important;
            margin-top: 0.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0,163,224,0.35) !important;
        }
        [data-testid="stButton"] button:hover {
            background-color: #38bdf8 !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(56,189,248,0.4) !important;
        }

        /* Password strength bar */
        .strength-bar {
            height: 4px; border-radius: 2px; margin-top: 4px;
            transition: width 0.4s ease, background 0.4s ease;
        }

        [data-testid="stAlert"] {
            background: rgba(220,38,38,0.12) !important;
            border: 1px solid rgba(220,38,38,0.25) !important;
            border-radius: 10px !important;
        }
        [data-testid="stAlert"] p { color: #fca5a5 !important; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.4, 1.8])

    with c1:
        st.markdown(
            '<div style="margin-bottom: 2rem;">'
            '<h2 style="color:#ffffff; font-weight:900; letter-spacing:3px; margin:0; font-size:1.6rem;">BARCLAYS<span style="color:#00A3E0;">.</span></h2>'
            '<div style="color:#00A3E0; font-size:0.75rem; font-weight:800; letter-spacing:1.5px; margin-top:1.5rem; text-transform:uppercase;">New Account Registration</div>'
            '<h1 style="color:#f8fafc; font-size:2.4rem; font-weight:900; margin:0.4rem 0 0.3rem 0; letter-spacing:-0.5px;">Create your ID.</h1>'
            '<p style="color:#94a3b8; font-size:0.9rem; margin:0; font-weight:500;">Register as a new analyst or admin below.</p>'
            "</div>",
            unsafe_allow_html=True,
        )

        full_name = st.text_input(
            "👤 Full Name", key="reg_name", placeholder="e.g. Priya Sharma"
        )
        email = st.text_input(
            "📧 Email Address", key="reg_email", placeholder="you@barclays.com"
        )

        # Role selector
        role = st.selectbox(
            "👔 Role",
            options=["Analyst", "Admin"],
            key="reg_role",
            help="Analysts see only their assigned customers. Admins can assign customers to analysts.",
        )

        password = st.text_input(
            "🔒 Password",
            type="password",
            key="reg_pass",
            placeholder="Min 8 chars, uppercase, digit, symbol",
        )
        password2 = st.text_input(
            "🔒 Confirm Password",
            type="password",
            key="reg_pass2",
            placeholder="Re-enter password",
        )

        # Password strength indicator
        if password:
            checks = [
                len(password) >= 8,
                any(c.isupper() for c in password),
                any(c.isdigit() for c in password),
                any(c in '!@#$%^&*(),.?":{}|<>' for c in password),
            ]
            strength = sum(checks)
            colors = ["#ef4444", "#f97316", "#eab308", "#22c55e"]
            labels = ["Weak", "Fair", "Good", "Strong"]
            st.markdown(
                f"""
                <div style="margin-top:-0.5rem; margin-bottom:0.8rem;">
                    <div style="background:#1e293b; border-radius:2px; height:4px;">
                        <div class="strength-bar" style="width:{strength * 25}%; background:{colors[strength - 1]}; height:4px; border-radius:2px;"></div>
                    </div>
                    <div style="color:{colors[strength - 1]}; font-size:0.72rem; font-weight:700; margin-top:4px; text-transform:uppercase; letter-spacing:1px;">
                        {"✓ " if strength == 4 else ""}{labels[strength - 1]}
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

        if st.button("🚀 Create Account & Enter"):
            if not full_name.strip():
                st.error("Please enter your full name.")
            elif not email.strip():
                st.error("Please enter your email address.")
            elif not password:
                st.error("Please choose a password.")
            elif password != password2:
                st.error("❌ Passwords do not match!")
            else:
                with st.spinner("Creating your account…"):
                    result = auth_db.register_user(
                        full_name.strip(), email.strip(), password, role
                    )
                    time.sleep(1.0)

                if result["ok"]:
                    # Auto-login after signup
                    user = auth_db.authenticate_user(email.strip(), password)
                    if user["ok"]:
                        st.session_state["app_state"] = "dashboard"
                        st.session_state["logged_in"] = True
                        st.session_state["user_id"] = user["id"]
                        st.session_state["full_name"] = user["full_name"]
                        st.session_state["email"] = user["email"]
                        st.session_state["role"] = user["role"]
                        st.session_state["username"] = user["email"]
                        st.session_state["analyst_name"] = user["full_name"]
                        st.session_state["assigned_role"] = user["role"]
                        st.success("✅ Account created! Redirecting to dashboard…")
                        time.sleep(0.8)
                        st.rerun()
                else:
                    st.error(f"❌ {result['error']}")

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        if st.button("⬅ Already have an account? Sign In"):
            st.session_state["app_state"] = "login"
            st.rerun()

        # Password rules hint
        st.markdown(
            """
            <div style="margin-top:1.5rem; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); 
                        border-radius:10px; padding:0.9rem 1rem; font-size:0.78rem; color:#64748b;">
                <div style="color:#38bdf8; font-weight:700; margin-bottom:0.5rem;">🔒 Password Requirements</div>
                <div>• At least 8 characters</div>
                <div>• One uppercase letter (A-Z)</div>
                <div>• One digit (0-9)</div>
                <div>• One special character (!@#$%...)</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            '<div style="display:flex; flex-direction:column; justify-content:center; height:100%; position:relative; z-index:5;">'
            '<div style="background:rgba(255,255,255,0.08); padding:0.4rem 1rem; border-radius:20px; color:#e0f2fe; font-size:0.78rem; font-weight:700; letter-spacing:1px; margin-bottom:1.5rem; display:inline-block; border:1px solid rgba(255,255,255,0.15);">PDIE • RISK INTELLIGENCE PLATFORM</div>'
            '<h1 style="color:#ffffff; font-size:4.5rem; font-weight:900; line-height:1.05; margin-bottom:1.5rem; letter-spacing:-1.5px; text-shadow: 0 4px 20px rgba(0,0,0,0.25);">Join the<br>vanguard of Risk.</h1>'
            '<p style="color:#e0f2fe; font-size:1.25rem; font-weight:400; max-width:85%; line-height:1.7; margin-bottom:3rem; opacity:0.9; border-left: 4px solid #4ade80; padding-left: 1rem;">Take control of high-liability portfolios with our industry-leading precision engines. Predict. Intervene. Protect.</p>'
            '<div style="display:flex; gap:2rem; flex-wrap:wrap;">'
            '<div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.15); padding:1.8rem; border-radius:16px; backdrop-filter:blur(12px); min-width:190px; box-shadow:0 10px 30px rgba(0,0,0,0.1);">'
            '<div style="color:#7dd3fc; font-size:2.5rem; font-weight:900; letter-spacing:-1px;">$2B+</div>'
            '<div style="color:#e2e8f0; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-top:0.3rem;">Assets Protected</div>'
            "</div>"
            '<div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.15); padding:1.8rem; border-radius:16px; backdrop-filter:blur(12px); min-width:190px; box-shadow:0 10px 30px rgba(0,0,0,0.1);">'
            '<div style="color:#4ade80; font-size:2.5rem; font-weight:900; letter-spacing:-1px;">5</div>'
            '<div style="color:#e2e8f0; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-top:0.3rem;">Analyst Teams</div>'
            "</div>"
            '<div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.15); padding:1.8rem; border-radius:16px; backdrop-filter:blur(12px); min-width:190px; box-shadow:0 10px 30px rgba(0,0,0,0.1);">'
            '<div style="color:#f9a8d4; font-size:2.5rem; font-weight:900; letter-spacing:-1px;">86.3%</div>'
            '<div style="color:#e2e8f0; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-top:0.3rem;">Model AUC</div>'
            "</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
