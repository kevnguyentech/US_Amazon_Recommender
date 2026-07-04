# verify_validation_fix.py
import sys
sys.path.insert(0, "api")
sys.path.insert(0, "model")

import main
from fastapi.testclient import TestClient

client = TestClient(main.app)

cases = [
    ("predict", {"user_idx": -1, "item_idx": 0}, 400),
    ("predict", {"user_idx": int(main.n_users) + 100, "item_idx": 0}, 400),
    ("predict", {"user_idx": 0, "item_idx": -1}, 400),
    ("predict", {"user_idx": 0, "item_idx": 0}, 200),
    ("recommend", {"user_idx": -1, "top_k": 5}, 400),
    ("recommend", {"user_idx": 0, "top_k": 0}, 400),
    ("recommend", {"user_idx": 0, "top_k": 5}, 200),
]

failures = []
for endpoint, payload, expected in cases:
    r = client.post(f"/{endpoint}", json=payload)
    ok = r.status_code == expected
    print(f"{'PASS' if ok else 'FAIL'} /{endpoint} {payload} -> {r.status_code} (expected {expected})")
    if not ok:
        failures.append((endpoint, payload))

if failures:
    print(f"\n{len(failures)} case(s) failed")
else:
    print("\nAll validation cases passed")