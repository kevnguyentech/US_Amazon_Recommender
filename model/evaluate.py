import pandas as pd
import torch
from two_tower import TwoTowerModel

full_df = pd.read_csv("data/amazon_reviews.csv")
full_df['user_idx'] = full_df['user_id'].astype('category').cat.codes
full_df['item_idx'] = full_df['item_id'].astype('category').cat.codes

n_users = full_df['user_idx'].max() + 1
n_items = full_df['item_idx'].max() + 1

test_df = pd.read_csv("data/test.csv")

model = TwoTowerModel(n_users, n_items, embedding_dim=16)
model.load_state_dict(torch.load("model/two_tower.pt"))
model.eval()

K = 5
RELEVANCE_THRESHOLD = 4.0  # rating >= 4 counts as "relevant"

precisions = []

import random

random_precisions = []
with torch.no_grad():
    for user_id, group in test_df.groupby('user_idx'):
        if len(group) < K:
            continue
        actual_ratings = group['rating'].values
        random_idx = random.sample(range(len(group)), K)
        random_actual = actual_ratings[random_idx]
        hits = (random_actual >= RELEVANCE_THRESHOLD).sum()
        random_precisions.append(hits / K)

avg_random = sum(random_precisions) / len(random_precisions)
print(f"Random baseline Precision@{K}: {avg_random:.4f}")

with torch.no_grad():
    for user_id, group in test_df.groupby('user_idx'):
        if len(group) < K:
            continue  # skip users with too few test items to rank meaningfully

        item_ids = torch.tensor(group['item_idx'].values, dtype=torch.long)
        user_ids = torch.full_like(item_ids, user_id)

        preds = model(user_ids, item_ids)
        top_k_idx = torch.topk(preds, K).indices

        actual_ratings = group['rating'].values
        top_k_actual = actual_ratings[top_k_idx.numpy()]

        hits = (top_k_actual >= RELEVANCE_THRESHOLD).sum()
        precisions.append(hits / K)

avg_precision = sum(precisions) / len(precisions)
print(f"Users evaluated: {len(precisions)}")
print(f"Precision@{K}: {avg_precision:.4f}")