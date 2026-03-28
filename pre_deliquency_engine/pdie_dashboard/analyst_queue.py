"""
analyst_queue.py — Analyst Queue Tab for PDIE Dashboard
=========================================================
  - Admin: sees ALL assignments + can assign/reassign any customer to any analyst
  - Analyst: sees only MY assignments + can self-request (claim) unassigned customers
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import auth_db


def _risk_badge(cat: str) -> str:
    colors = {
        "CRITICAL": ("#dc2626", "#fee2e2", "🔴"),
        "HIGH":     ("#f97316", "#ffedd5", "🟠"),
        "MEDIUM":   ("#ca8a04", "#fef9c3", "🟡"),
        "LOW":      ("#16a34a", "#dcfce7", "🟢"),
    }
    c, bg, icon = colors.get(cat, ("#64748b", "#f1f5f9", "⚪"))
    return (
        f'<span style="background:{bg}; color:{c}; border:1px solid {c}40; '
        f'padding:0.2rem 0.7rem; border-radius:20px; font-size:0.78rem; font-weight:700;">'
        f'{icon} {cat}</span>'
    )


def _status_badge(status: str) -> str:
    cfg = {
        "Active":   ("#16a34a", "#dcfce7", "●"),
        "Resolved": ("#1d4ed8", "#dbeafe", "✓"),
        "Escalated":("#dc2626", "#fee2e2", "⚠"),
    }
    c, bg, icon = cfg.get(status, ("#64748b", "#f1f5f9", "·"))
    return (
        f'<span style="background:{bg}; color:{c}; border:1px solid {c}40; '
        f'padding:0.15rem 0.6rem; border-radius:20px; font-size:0.75rem; font-weight:700;">'
        f'{icon} {status}</span>'
    )


def _render_table(rows: list, is_admin: bool):
    """Render the assignment table as styled HTML."""
    if not rows:
        st.info("No assignments match your filters.")
        return

    # Header
    headers = ["ID", "Customer ID", "Customer Name", "Risk Score", "Risk Level",
               "Analyst", "Assigned On", "Status"]
    grid = "60px 120px 1fr 100px 110px 140px 120px 110px"

    header_html = "".join(
        f'<div style="font-size:0.68rem; font-weight:700; color:#64748b; '
        f'text-transform:uppercase; letter-spacing:0.8px;">{h}</div>'
        for h in headers
    )
    st.markdown(f"""
        <div style="background:white; border-radius:16px; border:1px solid rgba(0,0,0,0.07);
                     overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.06);">
            <div style="display:grid; grid-template-columns:{grid};
                         gap:0.8rem; padding:0.8rem 1.2rem;
                         background:#f8fafc; border-bottom:1px solid #e2e8f0;">
                {header_html}
            </div>
    """, unsafe_allow_html=True)

    for row in rows:
        try:
            assigned_date = datetime.fromisoformat(row["assigned_at"]).strftime("%d %b %Y")
        except Exception:
            assigned_date = str(row.get("assigned_at", ""))[:10]

        score = float(row["risk_score"] or 0)
        score_color = (
            "#dc2626" if score >= 80 else
            "#f97316" if score >= 70 else
            "#ca8a04" if score >= 50 else "#16a34a"
        )

        st.markdown(f"""
            <div style="display:grid; grid-template-columns:{grid};
                         gap:0.8rem; padding:0.9rem 1.2rem; border-bottom:1px solid #f1f5f9;
                         align-items:center; background:white;"
                 onmouseover="this.style.background='#f8fafc'"
                 onmouseout="this.style.background='white'">
                <div style="font-size:0.8rem; color:#94a3b8; font-weight:600;">#{row['id']}</div>
                <div style="font-size:0.88rem; font-weight:700; color:#0f172a; font-family:monospace;">{row['customer_id']}</div>
                <div style="font-size:0.9rem; font-weight:500; color:#1e293b;">{row['customer_name'] or '—'}</div>
                <div style="font-size:0.9rem; font-weight:800; color:{score_color};">{score:.1f}</div>
                <div>{_risk_badge(row['risk_category'])}</div>
                <div style="font-size:0.85rem; font-weight:600; color:#00395D;">{row['analyst_name']}</div>
                <div style="font-size:0.8rem; color:#64748b;">{assigned_date}</div>
                <div>{_status_badge(row['status'])}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def show_analyst_queue(df=None):
    """Main entry point for the Analyst Queue tab."""

    role      = st.session_state.get("role", "Analyst")
    full_name = st.session_state.get("full_name", "User")
    email     = st.session_state.get("email", "")
    is_admin  = (role == "Admin")

    # ── Page Header ──────────────────────────────────────────────────
    subtitle = (
        "Full portfolio assignment view — manage & reassign customers"
        if is_admin else
        f"Customers assigned to you, {full_name.split()[0]}"
    )
    st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0f172a 0%,#00395D 60%,#00539B 100%);
                    padding:2rem 2.5rem; border-radius:18px; margin-bottom:2rem; color:white;
                    box-shadow:0 10px 40px rgba(0,57,93,0.4); position:relative; overflow:hidden;
                    border:1px solid rgba(255,255,255,0.12);">
            <div style="position:absolute;top:-40%;right:-10%;width:300px;height:300px;border-radius:50%;
                         background:radial-gradient(circle,rgba(255,255,255,0.08) 0%,transparent 70%);"></div>
            <div style="font-size:0.75rem;font-weight:700;letter-spacing:2px;color:#38bdf8;
                         text-transform:uppercase;margin-bottom:0.4rem;">
                {'🏢 Admin View — All Assignments' if is_admin else '📋 My Queue'}
            </div>
            <h1 style="margin:0 0 0.3rem 0;font-size:2rem;font-weight:800;letter-spacing:-0.5px;">
                Analyst Queue
            </h1>
            <p style="margin:0;color:#94a3b8;font-size:0.95rem;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Always read fresh from DB ─────────────────────────────────────
    # Do NOT cache this — we need live data across sessions
    if is_admin:
        all_assignments = auth_db.get_all_assignments()
    else:
        all_assignments = auth_db.get_my_assignments(email)

    summary = {
        "total":    len(all_assignments),
        "critical": sum(1 for a in all_assignments if a["risk_category"] == "CRITICAL"),
        "high":     sum(1 for a in all_assignments if a["risk_category"] == "HIGH"),
        "analysts": len(set(a["analyst_email"] for a in all_assignments)) if is_admin else 1,
    }

    # ── KPI Cards ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("📋", "Total Assigned",  str(summary["total"]),    "#00539B", c1),
        ("🔴", "Critical Cases",  str(summary["critical"]), "#dc2626", c2),
        ("🟠", "High Risk Cases", str(summary["high"]),     "#f97316", c3),
        ("👥", "Active Analysts" if is_admin else "Your Cases",
               str(summary["analysts"]) if is_admin else str(summary["total"]),
               "#16a34a", c4),
    ]
    for ic, label, val, color, col in kpis:
        with col:
            st.markdown(f"""
                <div style="background:white; border:1px solid rgba(0,0,0,0.07); border-radius:14px;
                             padding:1.2rem 1.4rem; box-shadow:0 2px 12px rgba(0,0,0,0.06);
                             border-top:3px solid {color};">
                    <div style="font-size:1.4rem; margin-bottom:0.3rem;">{ic}</div>
                    <div style="font-size:0.75rem; color:#64748b; font-weight:700;
                                 text-transform:uppercase; letter-spacing:0.8px;">{label}</div>
                    <div style="font-size:2rem; font-weight:900; color:#0f172a;
                                 letter-spacing:-1px; margin-top:0.2rem;">{val}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ADMIN: Assign Customer Panel ──────────────────────────────────
    if is_admin:
        with st.expander("➕ Assign Customer to Analyst", expanded=True):
            # Always fetch fresh analyst list from DB
            analysts = auth_db.get_all_analysts()

            if not analysts:
                st.warning("No users registered yet. Ask analysts to sign up first.")
            else:
                analyst_map = {
                    f"{a['full_name']}  ({a['role']})  —  {a['email']}": a
                    for a in analysts
                }

                # ── Customer picker from real dataset ──────────────────
                if df is not None and len(df) > 0:
                    # Build customer options from the loaded features df
                    # Include customer_id, name (if available), risk_score, risk_category
                    cust_df = df.copy()

                    # Already-assigned customer IDs
                    already_assigned_ids = {a["customer_id"] for a in auth_db.get_all_assignments()}

                    # Search bar to filter customers
                    search_cust = st.text_input(
                        "🔍 Search customer by ID or name",
                        placeholder="Type customer ID or name…",
                        key="aq_admin_search_cust"
                    )

                    # Filter dataset
                    name_col = "full_name" if "full_name" in cust_df.columns else None
                    if search_cust:
                        mask = cust_df["customer_id"].astype(str).str.contains(search_cust, case=False, na=False)
                        if name_col:
                            mask = mask | cust_df[name_col].astype(str).str.contains(search_cust, case=False, na=False)
                        cust_df = cust_df[mask]

                    # Only show top 200 by risk score to keep dropdown fast
                    if "risk_score" in cust_df.columns:
                        cust_df = cust_df.nlargest(200, "risk_score")

                    # Build label for each customer
                    def _cust_label(row):
                        cid  = str(row["customer_id"])
                        name = str(row.get(name_col, "")) if name_col else ""
                        score = f"{row['risk_score']:.0f}" if "risk_score" in row else "?"
                        cat  = str(row.get("risk_category", ""))
                        assigned = "✓ assigned" if cid in already_assigned_ids else ""
                        base = f"{cid}" + (f"  |  {name}" if name and name != "nan" else "") + f"  |  Risk {score}  |  {cat}"
                        return base + (f"  [{assigned}]" if assigned else "")

                    cust_options_labels = [_cust_label(row) for _, row in cust_df.iterrows()]
                    cust_options_rows   = [row for _, row in cust_df.iterrows()]

                    if not cust_options_labels:
                        st.info("No customers match your search. Try a different term.")
                    else:
                        col_pick, col_analyst = st.columns([3, 2])
                        with col_pick:
                            selected_cust_idx = st.selectbox(
                                "Select Customer from Dataset",
                                range(len(cust_options_labels)),
                                format_func=lambda i: cust_options_labels[i],
                                key="aq_admin_cust_pick"
                            )
                            selected_row = cust_options_rows[selected_cust_idx]
                            cust_id   = str(selected_row["customer_id"])
                            cust_name = str(selected_row.get(name_col, cust_id)) if name_col else cust_id
                            risk_score_val = float(selected_row.get("risk_score", 70.0))
                            risk_cat_val   = str(selected_row.get("risk_category", "HIGH"))

                            # Preview card
                            cat_colors = {"CRITICAL": "#dc2626", "HIGH": "#f97316", "MEDIUM": "#ca8a04", "LOW": "#16a34a"}
                            cat_col = cat_colors.get(risk_cat_val, "#64748b")
                            already = cust_id in already_assigned_ids
                            st.markdown(f"""
                                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
                                             padding:0.8rem 1rem; margin-top:0.3rem; font-size:0.85rem;">
                                    <div style="font-weight:700; color:#0f172a; font-size:0.95rem;">{cust_name}</div>
                                    <div style="color:#64748b; margin-top:2px; font-family:monospace; font-size:0.8rem;">{cust_id}</div>
                                    <div style="margin-top:6px;">
                                        <span style="background:{cat_col}20; color:{cat_col}; border:1px solid {cat_col}40;
                                                      padding:0.15rem 0.6rem; border-radius:20px; font-size:0.75rem; font-weight:700;">
                                            {risk_cat_val} · {risk_score_val:.1f}
                                        </span>
                                        {'<span style="margin-left:8px; background:#dbeafe; color:#1d4ed8; border:1px solid #93c5fd; padding:0.15rem 0.6rem; border-radius:20px; font-size:0.75rem; font-weight:700;">⟳ Will Reassign</span>' if already else ''}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                        with col_analyst:
                            selected_analyst_label = st.selectbox(
                                "Assign to Analyst",
                                list(analyst_map.keys()),
                                key="aq_analyst",
                                help="All registered users are listed"
                            )
                            sel_analyst = analyst_map[selected_analyst_label]
                            st.markdown(f"""
                                <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
                                             padding:0.8rem 1rem; margin-top:0.3rem; font-size:0.85rem;">
                                    <div style="font-weight:700; color:#15803d;">👤 {sel_analyst['full_name']}</div>
                                    <div style="color:#64748b; font-size:0.78rem; margin-top:2px;">{sel_analyst['email']}</div>
                                    <div style="color:#94a3b8; font-size:0.75rem; margin-top:2px;">{sel_analyst['role']}</div>
                                </div>
                            """, unsafe_allow_html=True)

                        if st.button("✅ Confirm Assignment", key="aq_btn_assign", use_container_width=True):
                            result = auth_db.assign_customer(
                                cust_id, cust_name, risk_score_val, risk_cat_val,
                                sel_analyst["email"], sel_analyst["full_name"],
                            )
                            if result.get("ok"):
                                action = "Reassigned" if already else "Assigned"
                                st.success(f"✅ {action} **{cust_name}** ({cust_id}) → **{sel_analyst['full_name']}**")
                                st.rerun()
                            else:
                                st.error(f"❌ {result.get('error', 'Unknown error')}")
                else:
                    # Fallback: df not loaded — manual entry
                    st.info("ℹ️ Customer dataset not loaded. Enter details manually:")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        fb_cust_id   = st.text_input("Customer ID", placeholder="e.g. C-2001", key="aq_fb_cust_id")
                        fb_cust_name = st.text_input("Customer Name", placeholder="e.g. Rajesh Kumar", key="aq_fb_cust_name")
                    with col_b:
                        fb_score = st.number_input("Risk Score (0–100)", 0.0, 100.0, 75.0, 0.5, key="aq_fb_score")
                        fb_cat   = st.selectbox("Risk Category", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], key="aq_fb_cat")

                    sel_fb = analyst_map[st.selectbox("Assign to Analyst", list(analyst_map.keys()), key="aq_fb_analyst")]
                    if st.button("✅ Confirm Assignment", key="aq_btn_fb_assign"):
                        if not fb_cust_id.strip():
                            st.error("❌ Please enter a Customer ID.")
                        else:
                            r = auth_db.assign_customer(fb_cust_id.strip(), fb_cust_name.strip(), fb_score, fb_cat,
                                                         sel_fb["email"], sel_fb["full_name"])
                            if r.get("ok"):
                                st.success(f"✅ **{fb_cust_id}** assigned to **{sel_fb['full_name']}**")
                                st.rerun()
                            else:
                                st.error(f"❌ {r.get('error')}")

    # ── ANALYST: Self-Assign (Claim) Panel ───────────────────────────
    else:
        with st.expander("🙋 Request / Claim a Customer Assignment", expanded=False):
            st.markdown("""
                <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
                             padding:0.8rem 1rem; margin-bottom:1rem; font-size:0.85rem; color:#1e40af;">
                    ℹ️ Enter the Customer ID your team lead shared with you and claim it as your assignment.
                </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                self_cust_id   = st.text_input("Customer ID", placeholder="e.g. C-5001", key="aq_self_cust_id")
                self_cust_name = st.text_input("Customer Name (optional)", placeholder="e.g. Amit Kumar", key="aq_self_cust_name")
            with col_b:
                self_risk_score = st.number_input("Risk Score (0–100)", 0.0, 100.0, 70.0, 0.5, key="aq_self_risk")
                self_risk_cat   = st.selectbox("Risk Category", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], key="aq_self_risk_cat")

            if st.button("🙋 Claim This Customer", key="aq_btn_self_assign"):
                if not self_cust_id.strip():
                    st.error("❌ Please enter a Customer ID.")
                else:
                    result = auth_db.assign_customer(
                        self_cust_id.strip(),
                        self_cust_name.strip() or self_cust_id.strip(),
                        float(self_risk_score),
                        self_risk_cat,
                        email,          # logged-in analyst's email
                        full_name,      # logged-in analyst's name
                    )
                    if result.get("ok"):
                        st.success(f"✅ Customer **{self_cust_id.strip()}** claimed! "
                                   f"It will now appear in your queue AND in the admin's view.")
                        st.rerun()
                    else:
                        st.error(f"❌ {result.get('error', 'Could not claim customer.')}")

    st.markdown("---")

    # ── Refresh hint ──────────────────────────────────────────────────
    rcol1, rcol2 = st.columns([5, 1])
    with rcol2:
        if st.button("🔄 Refresh", key="aq_refresh"):
            st.rerun()
    with rcol1:
        st.markdown(
            f"<div style='color:#64748b; font-size:0.82rem; padding-top:0.4rem;'>"
            f"Live view — {len(all_assignments)} total assignments loaded from database</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filter Bar ────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        search = st.text_input("Search", placeholder="🔍  Customer name or ID…",
                                key="aq_search", label_visibility="collapsed")
    with fc2:
        cat_filter = st.multiselect("Risk Filter", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                     default=[], key="aq_cat_filter",
                                     label_visibility="collapsed",
                                     placeholder="Filter by Risk Category")
    with fc3:
        status_filter = st.selectbox("Status", ["All", "Active", "Resolved", "Escalated"],
                                      key="aq_status_filter", label_visibility="collapsed")

    # Apply filters
    filtered = all_assignments
    if search:
        s = search.lower()
        filtered = [a for a in filtered
                    if s in a["customer_id"].lower()
                    or s in (a["customer_name"] or "").lower()
                    or s in (a["analyst_name"] or "").lower()]
    if cat_filter:
        filtered = [a for a in filtered if a["risk_category"] in cat_filter]
    if status_filter != "All":
        filtered = [a for a in filtered if a["status"] == status_filter]

    st.markdown(
        f"<div style='color:#64748b; font-size:0.85rem; margin-bottom:0.8rem;'>"
        f"Showing <strong>{len(filtered)}</strong> of <strong>{len(all_assignments)}</strong> assignments</div>",
        unsafe_allow_html=True
    )

    if not filtered:
        st.info("No customers found matching your filters.")
    else:
        _render_table(filtered, is_admin)

    # ── Update Status ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔄 Update Case Status / Add Notes"):
        st.markdown("""
            <div style="background:#fef9c3; border:1px solid #fde68a; border-radius:8px;
                         padding:0.7rem 1rem; margin-bottom:0.8rem; font-size:0.82rem; color:#92400e;">
                💡 Use the <strong>ID #</strong> column from the table above to update a case status.
            </div>
        """, unsafe_allow_html=True)

        upd_col1, upd_col2 = st.columns(2)
        with upd_col1:
            upd_id     = st.number_input("Assignment ID (#)", min_value=1, step=1, key="aq_upd_id")
            upd_status = st.selectbox("New Status", ["Active", "Resolved", "Escalated"], key="aq_upd_status")
        with upd_col2:
            upd_notes = st.text_area("Notes / Comments", placeholder="Add case notes here…",
                                      height=100, key="aq_upd_notes")

        if st.button("💾 Save Update", key="aq_btn_update"):
            result = auth_db.update_assignment_status(int(upd_id), upd_status, upd_notes)
            if result.get("ok"):
                st.success(f"✅ Assignment #{upd_id} → **{upd_status}**")
                st.rerun()
            else:
                st.error(f"❌ {result.get('error', 'Update failed.')}")

    # ── Charts (Admin only) ───────────────────────────────────────────
    if is_admin and all_assignments:
        st.markdown("---")
        st.markdown("##### 📊 Queue Analytics")

        ch1, ch2 = st.columns(2)
        with ch1:
            by_analyst: dict = {}
            for a in all_assignments:
                by_analyst[a["analyst_name"]] = by_analyst.get(a["analyst_name"], 0) + 1

            fig = go.Figure(go.Bar(
                x=list(by_analyst.values()),
                y=list(by_analyst.keys()),
                orientation="h",
                marker_color="#00539B",
                text=list(by_analyst.values()),
                textposition="auto",
            ))
            fig.update_layout(
                title="Assignments per Analyst",
                height=max(250, len(by_analyst) * 50),
                margin=dict(t=40, b=20, l=20, r=20),
                plot_bgcolor="rgba(248,250,252,0.5)",
                xaxis_title="Cases",
            )
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            by_cat: dict = {}
            for a in all_assignments:
                by_cat[a["risk_category"]] = by_cat.get(a["risk_category"], 0) + 1

            cat_colors = {"CRITICAL": "#dc2626", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}
            fig2 = go.Figure(go.Pie(
                labels=list(by_cat.keys()),
                values=list(by_cat.values()),
                hole=0.5,
                marker_colors=[cat_colors.get(k, "#94a3b8") for k in by_cat],
                textinfo="label+percent",
            ))
            fig2.update_layout(
                title="Queue by Risk Category",
                height=300,
                margin=dict(t=40, b=20, l=20, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
