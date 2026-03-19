"""
Streamlit Page: Register Reference Signatures (single or bulk)
"""
import httpx
import streamlit as st

st.set_page_config(page_title="Register Signature", page_icon="📋")
st.title("📋 Register Reference Signatures")
st.markdown(
    "Upload **one or more** genuine signature images to enroll them as references for a user. "
    "For best verification accuracy, register **5–10 signatures** capturing natural variation."
)

api = st.session_state.get("api_base_url", "http://localhost:8000")
user_id = st.session_state.get("user_id")

user_email = st.session_state.get("user_email") or f"User {user_id}"
st.info(f"Registering signatures for: **{user_email}**")

# ─── Registration form ────────────────────────────────────────────────────────
with st.form("register_form"):
    uploaded_files = st.file_uploader(
        "Upload Signature Image(s)",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        help=(
            "Select multiple files at once (Ctrl+Click / Cmd+Click). "
            "Each file is registered as a separate reference. "
            "Labels are auto-generated as ref_01, ref_02, … unless you provide a prefix below."
        ),
    )

    col_prefix, col_start = st.columns([2, 1])
    with col_prefix:
        label_prefix = st.text_input(
            "Label prefix (optional)",
            value="ref",
            placeholder="e.g. ref, sample, genuine",
            help="Labels will be: <prefix>_01, <prefix>_02, …  Leave as 'ref' for defaults.",
        )
    with col_start:
        label_start = st.number_input(
            "Start index",
            min_value=1,
            max_value=99,
            value=1,
            step=1,
            help="Start numbering from this index (useful when adding to existing references).",
        )

    submitted = st.form_submit_button("📤 Register All Signatures", use_container_width=True, type="primary")

# ─── Submission handler ───────────────────────────────────────────────────────
if submitted:
    if not uploaded_files:
        st.error("⛔ Please upload at least one signature image.")
        st.stop()

    total = len(uploaded_files)
    st.markdown(f"**Registering {total} signature(s) for User ID {user_id}…**")
    progress = st.progress(0, text="Starting…")

    results = []   # list of dicts for the summary table

    for idx, uploaded in enumerate(uploaded_files, start=0):
        label = f"{label_prefix.strip() or 'ref'}_{int(label_start) + idx:02d}"
        progress.progress((idx) / total, text=f"Uploading {uploaded.name} → `{label}`…")

        try:
            file_bytes = uploaded.read()
            resp = httpx.post(
                f"{api}/api/signatures/register",
                files={"file": (uploaded.name, file_bytes, uploaded.type)},
                data={"user_id": str(user_id), "label": label},
                timeout=30,
            )

            if resp.status_code == 201:
                r = resp.json()
                results.append({
                    "File": uploaded.name,
                    "Label": label,
                    "Signature ID": r.get("signature_id", "—"),
                    "Status": "✅ Registered",
                })
            elif resp.status_code == 409:
                results.append({
                    "File": uploaded.name,
                    "Label": label,
                    "Signature ID": "—",
                    "Status": f"⚠️ Duplicate label (already exists)",
                })
            else:
                detail = resp.json().get("detail", resp.text)
                results.append({
                    "File": uploaded.name,
                    "Label": label,
                    "Signature ID": "—",
                    "Status": f"❌ {resp.status_code}: {detail}",
                })

        except httpx.ConnectError:
            results.append({
                "File": uploaded.name,
                "Label": label,
                "Signature ID": "—",
                "Status": "❌ Cannot reach API",
            })
        except Exception as exc:
            results.append({
                "File": uploaded.name,
                "Label": label,
                "Signature ID": "—",
                "Status": f"❌ Error: {exc}",
            })

    progress.progress(1.0, text="Done.")

    # ── Summary table ──────────────────────────────────────────────────────────
    ok_count  = sum(1 for r in results if r["Status"].startswith("✅"))
    err_count = total - ok_count

    if ok_count == total:
        st.success(f"🎉 All {total} signature(s) registered successfully!")
    elif ok_count > 0:
        st.warning(f"⚠️ {ok_count}/{total} registered. {err_count} failed — see table below.")
    else:
        st.error(f"❌ All {total} registration(s) failed.")

    st.table(results)

    if ok_count > 0:
        st.info(
            f"💡 User ID **{user_id}** now has reference signatures enrolled. "
            "Go to **Verify** to test a query signature against them."
        )

# ─── Existing signatures viewer ───────────────────────────────────────────────
st.divider()
st.subheader("📂 Existing Reference Signatures")

# Session state for persistence
if "signatures_data" not in st.session_state:
    st.session_state.signatures_data = None

def fetch_signatures():
    """Fetch current user's signatures from API and update session state."""
    try:
        r = httpx.get(f"{api}/api/signatures/{user_id}", timeout=10)
        if r.status_code == 200:
            st.session_state.signatures_data = r.json()
        else:
            st.error(f"Could not fetch signatures (HTTP {r.status_code}).")
            st.session_state.signatures_data = None
    except Exception as exc:
        st.error(f"Could not load signatures: {exc}")
        st.session_state.signatures_data = None

# Auto-fetch if user_id is set but we have no data yet
if user_id and st.session_state.signatures_data is None:
    with st.spinner("Fetching enrolled signatures…"):
        fetch_signatures()

col_refresh, _ = st.columns([1, 3])
with col_refresh:
    if st.button("🔄 Refresh List", use_container_width=True):
        fetch_signatures()

data = st.session_state.signatures_data

if data:
    st.metric("Total Enrolled Signatures", data["total"])

    if data["total"] == 0:
        st.info("No signatures registered yet for this user.")
    else:
        # ── Bulk delete ───────────────────────────────────────────────
        st.markdown("**⚠️ Danger Zone**")
        confirm_delete_all = st.checkbox(
            f"I want to delete ALL {data['total']} signature(s) for User ID {user_id}",
            key="confirm_delete_all",
        )
        if confirm_delete_all:
            if st.button(
                f"🗑️ Delete All {data['total']} Signature(s)",
                type="primary",
                use_container_width=True,
                key="btn_delete_all",
            ):
                failed = []
                with st.spinner("Deleting all signatures…"):
                    for sig in data["signatures"]:
                        try:
                            dr = httpx.delete(f"{api}/api/signatures/{sig['id']}", timeout=10)
                            if dr.status_code != 204:
                                failed.append(sig["id"])
                        except Exception:
                            failed.append(sig["id"])
                
                if not failed:
                    st.success(f"✅ All {data['total']} signatures deleted.")
                    st.session_state.signatures_data = None # Trigger refresh
                    st.rerun()
                else:
                    st.error(f"⚠️ {len(failed)} deletion(s) failed.")
                    fetch_signatures() # Refresh to see remaining
                    st.rerun()

        st.divider()

        # ── Individual signature cards ────────────────────────────────
        for sig in data["signatures"]:
            with st.expander(f"Signature #{sig['id']} — `{sig.get('label') or 'No label'}`"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Signature ID:** {sig['id']}")
                    st.write(f"**Label:** {sig.get('label') or '—'}")
                with col_b:
                    st.write(f"**Created:** {str(sig.get('created_at', '—'))[:19].replace('T', ' ')}")
                    st.write(f"**File:** `{sig.get('file_path', '—')}`")

                if st.button(f"🗑️ Delete #{sig['id']}", key=f"del_{sig['id']}"):
                    with st.spinner(f"Deleting signature #{sig['id']}…"):
                        try:
                            dr = httpx.delete(f"{api}/api/signatures/{sig['id']}", timeout=10)
                            if dr.status_code == 204:
                                st.success(f"Signature #{sig['id']} deleted.")
                                fetch_signatures() # Refresh session state
                                st.rerun()
                            else:
                                st.error("Delete failed.")
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
else:
    if user_id:
        st.info("Click 'Refresh List' to view signatures for this user.")
