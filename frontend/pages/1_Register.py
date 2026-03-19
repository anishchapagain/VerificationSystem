"""
Streamlit Page: Register Reference Signature
"""
import httpx
import streamlit as st

st.set_page_config(page_title="Register Signature", page_icon="📋")
st.title("📋 Register Reference Signature")
st.markdown("Upload a clear handwritten signature image to enroll it as a reference for a user.")

api = st.session_state.get("api_base_url", "http://localhost:8000")
user_id = st.session_state.get("user_id")

if not user_id:
    st.warning("⚠️ Please set your User ID on the Home page first.")
    st.stop()

st.info(f"Registering signature for **User ID: {user_id}**")

with st.form("register_form"):
    uploaded = st.file_uploader(
        "Upload Signature Image",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Scan or photograph of the handwritten signature.",
    )
    label = st.text_input("Label (optional)", placeholder="e.g. Primary, Backup")
    submitted = st.form_submit_button("📤 Register Signature", use_container_width=True)

if submitted:
    if uploaded is None:
        st.error("Please upload a signature image.")
    else:
        with st.spinner("Processing and registering..."):
            try:
                files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}
                data  = {"user_id": str(user_id)}
                if label:
                    data["label"] = label

                resp = httpx.post(
                    f"{api}/api/signatures/register",
                    files=files, data=data, timeout=30
                )
                if resp.status_code == 201:
                    result = resp.json()
                    st.success(f"✅ Signature registered! ID: **{result['signature_id']}**")
                    st.json(result)
                else:
                    st.error(f"Registration failed ({resp.status_code}): {resp.json().get('detail')}")
            except Exception as e:
                st.error(f"Request failed: {e}")

st.divider()
st.subheader("📂 Existing Reference Signatures")
if st.button("Refresh List"):
    try:
        r = httpx.get(f"{api}/api/signatures/{user_id}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            st.metric("Total Signatures", data["total"])
            for sig in data["signatures"]:
                with st.expander(f"Signature #{sig['id']} — {sig.get('label') or 'No label'}"):
                    st.write(f"**Created:** {sig['created_at']}")
                    st.write(f"**File:** `{sig['file_path']}`")
                    if st.button(f"🗑️ Delete #{sig['id']}", key=f"del_{sig['id']}"):
                        dr = httpx.delete(f"{api}/api/signatures/{sig['id']}", timeout=10)
                        if dr.status_code == 204:
                            st.success("Deleted.")
                        else:
                            st.error("Delete failed.")
    except Exception as e:
        st.error(f"Could not load signatures: {e}")
