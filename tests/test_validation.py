import pytest
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


@pytest.mark.parametrize("endpoint,payload,expected", cases)
def test_endpoint_validation(endpoint, payload, expected):
    r = client.post(f"/{endpoint}", json=payload)
    assert r.status_code == expected