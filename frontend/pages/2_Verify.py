"""
Streamlit Page: Verify Signature
"""
import httpx
import streamlit as st

st.set_page_config(page_title="Verify Signature", page_icon="🔍")
st.title("🔍 Verify Signature")
st.markdown("Upload a query signature (image or video) to verify it against stored references.")

api     = st.session_state.get("api_base_url", "http://localhost:8000")
user_id = st.session_state.get("user_id")

user_email = st.session_state.get("user_email") or f"User {user_id}"
st.info(f"Verifying against references for: **{user_email}**")

# ── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Verification Settings")
    st.markdown("Customize how the system decides a match.")
    
    threshold_val = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
        help="Higher = Strict, Lower = Permissive."
    )
    
    strategy_label = st.selectbox(
        "Match Strategy",
        options=["Highest Similarity", "Lowest Similarity", "Average Similarity"],
        index=0,
        help="How to aggregate multiple reference signatures."
    )
    
    # Map label to backend value
    strategy_map = {
        "Highest Similarity": "highest",
        "Lowest Similarity": "lowest",
        "Average Similarity": "average"
    }
    match_strategy = strategy_map[strategy_label]

with st.form("verify_form"):
    uploaded = st.file_uploader(
        "Upload Query Signature",
        type=["png", "jpg", "jpeg", "bmp", "mp4", "avi", "mov"],
        help="Image or video of the signature to verify.",
    )
    submitted = st.form_submit_button("🔍 Verify Now", use_container_width=True)

if submitted:
    if uploaded is None:
        st.error("Please upload a file.")
    else:
        with st.spinner("Verifying signature..."):
            try:
                # Read file once
                file_bytes = uploaded.read()
                files = {"file": (uploaded.name, file_bytes, uploaded.type)}
                data  = {
                    "user_id": str(user_id),
                    "threshold": str(threshold_val),
                    "match_strategy": match_strategy
                }
                resp  = httpx.post(
                    f"{api}/api/signatures/verify",
                    files=files, data=data, timeout=60
                )
                if resp.status_code == 200:
                    r = resp.json()
                    verdict = r["verdict"]

                    # ── Visual verdict ─────────────────────────────────────
                    if verdict:
                        st.success(f"## ✅ MATCH — {r['confidence']} Confidence")
                    else:
                        st.error(f"## ❌ NO MATCH — {r['confidence']} Confidence")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Score", f"{r['score']:.4f}")
                    col2.metric("Threshold", f"{r['threshold_used']}")
                    col3.metric("Strategy", r["match_strategy"].capitalize())
                    col4.metric("Source", r["source"].capitalize())

                    st.divider()
                    st.subheader("Score Breakdown")
                    for item in r.get("score_breakdown", []):
                        bar_val = min(max(item["score"], 0.0), 1.0)
                        st.write(f"Signature #{item['signature_id']}: `{item['score']:.4f}`")
                        st.progress(bar_val)

                    with st.expander("Raw JSON Response"):
                        st.json(r)
                else:
                    st.error(f"Verification failed ({resp.status_code}): {resp.json().get('detail')}")
            except Exception as e:
                st.error(f"Request failed: {e}")
