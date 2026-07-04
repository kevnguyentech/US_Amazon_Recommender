import pandas as pd
import main


def test_recommendations_are_trained_items():
    train_df = pd.read_csv("data/train.csv")
    valid_ids = set(train_df["item_idx"].unique())

    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    for user_idx in range(30):
        r = client.post("/recommend", json={"user_idx": user_idx, "top_k": 5})
        if r.status_code != 200:
            continue
        recs = [x["item_idx"] for x in r.json()["recommendations"]]
        bad = [i for i in recs if i not in valid_ids]
        assert not bad, f"user {user_idx} got untrained items: {bad}"