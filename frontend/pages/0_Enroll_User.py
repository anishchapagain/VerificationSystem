"""
Streamlit Page: Enroll New Client User
========================================
Module  : frontend/pages/0_Enroll_User.py
Purpose : Allow operators to register a new client in the system.
          Calls POST /api/users/register, checks for duplicate emails,
          validates all fields, and displays the generated User ID on success.
"""

import re

import httpx
import streamlit as st

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Enroll User", page_icon="👤", layout="centered")

api = st.session_state.get("api_base_url", "http://localhost:8000")

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("👤 Enroll New Client")
st.markdown(
    "Register a new client in the system. "
    "A unique **User ID** will be generated and displayed after successful enrollment. "
    "Use that ID when registering or verifying their signatures."
)
st.divider()

# ─── Validation helpers ───────────────────────────────────────────────────────

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
PASSWORD_MIN_LEN = 8


def _validate_name(name: str) -> str | None:
    """Return an error message or None if valid."""
    name = name.strip()
    if not name:
        return "Full name is required."
    if len(name) < 2:
        return "Name must be at least 2 characters."
    if len(name) > 100:
        return "Name must be 100 characters or fewer."
    if not re.match(r"^[A-Za-z\s'\-\.]+$", name):
        return "Name may only contain letters, spaces, hyphens, apostrophes, or periods."
    return None


def _validate_email(email: str) -> str | None:
    email = email.strip()
    if not email:
        return "Email address is required."
    if not EMAIL_REGEX.match(email):
        return "Please enter a valid email address."
    return None


def _validate_password(password: str) -> str | None:
    if not password:
        return "Password is required."
    if len(password) < PASSWORD_MIN_LEN:
        return f"Password must be at least {PASSWORD_MIN_LEN} characters."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[0-9]", password):
        return "Password must contain at least one digit."
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?]", password):
        return "Password must contain at least one special character."
    return None


# ─── Registration form ────────────────────────────────────────────────────────
with st.form("enroll_user_form", clear_on_submit=False):
    st.subheader("Client Details")

    name = st.text_input(
        "Full Name *",
        placeholder="e.g. Jane Smith",
        help="Letters, spaces, hyphens, and apostrophes only.",
    )
    email = st.text_input(
        "Email Address *",
        placeholder="e.g. jane.smith@company.com",
        help="Must be unique — used to detect if the client already exists.",
    )

    st.markdown("**Password** *(min 8 chars, 1 uppercase, 1 digit, 1 special character)*")
    col_pw, col_cpw = st.columns(2)
    with col_pw:
        password = st.text_input("Password *", type="password", placeholder="••••••••••")
    with col_cpw:
        confirm_password = st.text_input("Confirm Password *", type="password", placeholder="••••••••••")

    submitted = st.form_submit_button("✅ Enroll Client", use_container_width=True, type="primary")

# ─── Form submission logic ────────────────────────────────────────────────────
if submitted:
    # --- Client-side validation ---
    errors = []

    name_err = _validate_name(name)
    if name_err:
        errors.append(name_err)

    email_err = _validate_email(email)
    if email_err:
        errors.append(email_err)

    pw_err = _validate_password(password)
    if pw_err:
        errors.append(pw_err)

    if not errors and password != confirm_password:
        errors.append("Passwords do not match.")

    if errors:
        for err in errors:
            st.error(f"⛔ {err}")
        st.stop()

    # --- API call ---
    with st.spinner("Enrolling client…"):
        try:
            resp = httpx.post(
                f"{api}/api/users/register",
                json={
                    "name": name.strip(),
                    "email": email.strip().lower(),
                    "password": password,
                },
                timeout=15,
                headers={"Content-Type": "application/json", "accept": "application/json"},
            )
        except httpx.ConnectError:
            st.error("❌ Cannot reach the API. Make sure the backend is running.")
            st.stop()
        except httpx.TimeoutException:
            st.error("❌ Request timed out. The server may be overloaded.")
            st.stop()
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")
            st.stop()

    # --- Handle response ---
    if resp.status_code == 201:
        result = resp.json()
        user_id = result.get("id")

        st.success("🎉 Client enrolled successfully!")

        # Highlight the generated User ID prominently
        st.markdown("---")
        st.markdown("### 🪪 Generated User ID")

        id_col, detail_col = st.columns([1, 2])
        with id_col:
            st.metric(label="User ID", value=user_id)
            st.caption("Share this ID with the operator for signature registration.")
        with detail_col:
            st.markdown(f"**Name:** {result.get('name')}")
            st.markdown(f"**Email:** {result.get('email')}")
            st.markdown(f"**Registered:** {result.get('created_at', '—')[:19].replace('T', ' ')}")

        # Store in session so the operator can immediately navigate to Register
        st.session_state.user_id = user_id
        st.info(
            f"💡 **User ID {user_id}** has been saved to your session. "
            "You can now go to **Register Signature** to enroll their signature."
        )

        with st.expander("📄 Full API Response"):
            st.json(result)

    elif resp.status_code == 409:
        # Duplicate email — the backend raises 409 for UserAlreadyExistsError
        detail = resp.json().get("detail", "")
        st.error("⚠️ A client with this email address already exists.")
        st.markdown(
            "If you need their User ID, please check the database or ask your administrator. "
            f"\n\n*Server message:* `{detail}`"
        )

    elif resp.status_code == 422:
        # Pydantic validation error from the backend
        detail = resp.json().get("detail", [])
        st.error("⛔ Validation error returned by the server:")
        if isinstance(detail, list):
            for item in detail:
                loc = " → ".join(str(x) for x in item.get("loc", []))
                msg = item.get("msg", "")
                st.markdown(f"- **{loc}**: {msg}")
        else:
            st.code(str(detail))

    else:
        st.error(f"❌ Registration failed (HTTP {resp.status_code}).")
        try:
            st.code(resp.json())
        except Exception:
            st.code(resp.text)

# ─── Divider + lookup section ─────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Look Up Existing Client")
st.markdown("If a client is already enrolled, enter their User ID below to view their profile.")

with st.form("lookup_form"):
    lookup_id = st.number_input("User ID", min_value=1, step=1, value=1)
    lookup_btn = st.form_submit_button("🔎 Look Up", use_container_width=True)

if lookup_btn:
    with st.spinner("Fetching profile…"):
        try:
            r = httpx.get(f"{api}/api/users/me/{int(lookup_id)}", timeout=10)
        except Exception as exc:
            st.error(f"❌ Request failed: {exc}")
            st.stop()

    if r.status_code == 200:
        u = r.json()
        st.success("✅ Client found.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("User ID", u.get("id"))
            st.markdown(f"**Name:** {u.get('name')}")
        with col2:
            st.markdown(f"**Email:** {u.get('email')}")
            st.markdown(f"**Registered:** {str(u.get('created_at', '—'))[:19].replace('T', ' ')}")

        # Save to session for quick navigation
        if st.button(f"📌 Set User ID to {u['id']} for this session"):
            st.session_state.user_id = u["id"]
            st.success(f"Session User ID updated to {u['id']}.")
    elif r.status_code == 404:
        st.warning(f"⚠️ No client found with User ID **{lookup_id}**.")
    else:
        st.error(f"❌ Lookup failed (HTTP {r.status_code}).")
