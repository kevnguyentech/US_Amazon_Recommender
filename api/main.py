import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from two_tower import TwoTowerModel

app = FastAPI()

full_df = pd.read_csv("data/amazon_reviews.csv")
full_df['user_idx'] = full_df['user_id'].astype('category').cat.codes
full_df['item_idx'] = full_df['item_id'].astype('category').cat.codes

n_users = full_df['user_idx'].max() + 1
n_items = full_df['item_idx'].max() + 1

model = TwoTowerModel(n_users, n_items, embedding_dim=16)
model.load_state_dict(torch.load("model/two_tower.pt"))
model.eval()

class PredictRequest(BaseModel):
    user_idx: int
    item_idx: int

@app.post("/predict")
def predict(req: PredictRequest):
    with torch.no_grad():
        user_tensor = torch.tensor([req.user_idx], dtype=torch.long)
        item_tensor = torch.tensor([req.item_idx], dtype=torch.long)
        pred = model(user_tensor, item_tensor)
    return {"predicted_rating": float(pred.item())}

class RecommendRequest(BaseModel):
    user_idx: int
    top_k: int = 10

@app.post("/recommend")
def recommend(req: RecommendRequest):
    all_items = torch.arange(n_items, dtype=torch.long)
    user_tensor = torch.full_like(all_items, req.user_idx)

    with torch.no_grad():
        preds = model(user_tensor, all_items)

    top_k = torch.topk(preds, req.top_k)
    top_items = top_k.indices.tolist()
    top_scores = top_k.values.tolist()

    return {
        "user_idx": req.user_idx,
        "recommendations": [
            {"item_idx": item, "predicted_rating": score}
            for item, score in zip(top_items, top_scores)
        ]
    }