import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

API = "http://localhost:8000"

# Define test cases manually
# (register_image, verify_image, expected_verdict, description)
tests = [
    # Same person — should MATCH
    ("data/my_signatures/genuine/user_056_sig_01.png",
     "data/my_signatures/genuine/user_056_sig_08.png",
     True, "Same person genuine"),

    # Forged — should NO MATCH
    ("data/my_signatures/genuine/user_056_sig_01.png",
     "data/my_signatures/forged/user_056_forg_01.png",
     False, "Forged signature"),

    # Different person — should NO MATCH
    ("data/my_signatures/genuine/user_056_sig_01.png",
     "data/my_signatures/genuine/user_057_sig_01.png",
     False, "Different person"),

    # Same person signed on different day — should MATCH
    ("data/my_signatures/genuine/user_056_sig_01.png",
     "data/my_signatures/genuine/user_056_sig_12.png",
     True, "Same person different day"),
]

# Register user
r = httpx.post(f"{API}/api/users/register",
    json={"name": "Test", "email": "test@test.com", "password": "Test123!"})
user_id = r.json()["id"]

passed = 0
failed = 0

for reg_img, ver_img, expected, desc in tests:
    # Clean slate — register fresh
    httpx.delete(f"{API}/api/users/{user_id}")
    r = httpx.post(f"{API}/api/users/register",
        json={"name": "Test", "email": f"t{desc}@test.com", "password": "Test123!"})
    uid = r.json()["id"]

    with open(reg_img, "rb") as f:
        httpx.post(f"{API}/api/signatures/register",
            files={"file": f}, data={"user_id": uid})

    with open(ver_img, "rb") as f:
        r = httpx.post(f"{API}/api/signatures/verify",
            files={"file": f}, data={"user_id": uid})
    result = r.json()

    ok = result["verdict"] == expected
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok: passed += 1
    else:  failed += 1

    print(f"{status} | {desc:30s} | score={result['score']:.4f} | verdict={result['verdict_label']}")

print(f"\nResult: {passed}/{len(tests)} passed")