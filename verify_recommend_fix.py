# verify_recommend_fix.py
import sys
sys.path.insert(0, "api")
sys.path.insert(0, "model")

import pandas as pd
import main
from fastapi.testclient import TestClient

client = TestClient(main.app)
train_df = pd.read_csv("data/train.csv")
valid_ids = set(train_df["item_idx"].unique())

n_users_to_check = 30
failures = []

for u in range(n_users_to_check):
    r = client.post("/recommend", json={"user_idx": u, "top_k": 5})
    if r.status_code != 200:
        continue
    recs = [x["item_idx"] for x in r.json()["recommendations"]]
    bad = [i for i in recs if i not in valid_ids]
    if bad:
        failures.append((u, bad))

if failures:
    print(f"FAIL: {len(failures)} users got untrained items")
    for u, bad in failures[:5]:
        print(f"  user {u}: {bad}")
else:
    print(f"PASS: all recommendations across {n_users_to_check} users are trained items")