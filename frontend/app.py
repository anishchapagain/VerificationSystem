"""
Streamlit Frontend — Main Entry Point
========================================
Module  : frontend/app.py
Purpose : Configure the multi-page Streamlit app and shared session state.

Run:
    streamlit run frontend/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Signature Verifier",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Shared session defaults ──────────────────────────────────────────────────
if "api_base_url" not in st.session_state:
    import os
    st.session_state.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "token" not in st.session_state:
    st.session_state.token = None

# ─── Home Page ────────────────────────────────────────────────────────────────
st.title("✍️ Signature Verifier")
st.markdown("### AI-Powered Handwritten Signature Authentication")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**📋 Register**\nUpload a reference signature to enroll a user.")
with col2:
    st.success("**🔍 Verify**\nUpload a query signature and get a MATCH verdict.")
with col3:
    st.warning("**📊 History**\nView the verification audit log with scores.")

st.markdown("---")
st.markdown(
    "Use the **sidebar** to navigate between pages. "
    "Start by setting your **User ID** in the sidebar, then register a reference signature."
)

# ─── Sidebar: Quick Config ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    user_id_input = st.number_input(
        "Your User ID", min_value=1, step=1,
        value=st.session_state.user_id or 1,
        help="Set this before registering or verifying signatures.",
    )
    if st.button("Set User ID"):
        st.session_state.user_id = int(user_id_input)
        st.success(f"User ID set to {st.session_state.user_id}")

    st.divider()
    st.caption(f"API: `{st.session_state.api_base_url}`")

    # Health check
    if st.button("🏥 Check API Health"):
        import httpx
        try:
            r = httpx.get(f"{st.session_state.api_base_url}/health", timeout=5)
            data = r.json()
            if data.get("status") == "healthy":
                st.success(f"API healthy | v{data.get('version')}")
            else:
                st.warning(f"API degraded | db={data.get('database')}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
