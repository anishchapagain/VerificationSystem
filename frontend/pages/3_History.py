"""
Streamlit Page: Verification History
"""
import httpx
import streamlit as st

st.set_page_config(page_title="Verification History", page_icon="📊")
st.title("📊 Verification History")
st.markdown("Audit log of all signature verification events for the current user.")

api     = st.session_state.get("api_base_url", "http://localhost:8000")
user_id = st.session_state.get("user_id")

if not user_id:
    st.warning("⚠️ Please set your User ID on the Home page first.")
    st.stop()

user_email = st.session_state.get("user_email") or f"User {user_id}"
st.info(f"Viewing history for: **{user_email}**")

limit  = st.slider("Records to load", min_value=10, max_value=200, value=50, step=10)
offset = st.number_input("Offset", min_value=0, value=0, step=10)

if st.button("🔄 Load History", use_container_width=True):
    with st.spinner("Loading..."):
        try:
            r = httpx.get(
                f"{api}/api/signatures/history/{user_id}",
                params={"limit": limit, "offset": offset},
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                logs = data.get("logs", [])
                st.metric("Records returned", data["total_returned"])

                if not logs:
                    st.info("No verification history yet.")
                else:
                    for log in logs:
                        verdict_icon = "✅" if log["verdict"] else "❌"
                        with st.expander(
                            f"{verdict_icon} Log #{log['id']} — Score: {log['score']:.4f} — {log['created_at'][:19]}"
                        ):
                            col1, col2, col3 = st.columns(3)
                            col1.write(f"**Verdict:** {log['verdict_label']}")
                            col2.write(f"**Score:** `{log['score']:.6f}`")
                            col3.write(f"**Source:** {log['source'].capitalize()}")
                            st.write(f"**Threshold used:** `{log['threshold_used']}`")
                            if log.get("best_match_id"):
                                st.write(f"**Best match sig ID:** `{log['best_match_id']}`")
            else:
                st.error(f"Failed ({r.status_code}): {r.json().get('detail')}")
        except Exception as e:
            st.error(f"Request failed: {e}")
