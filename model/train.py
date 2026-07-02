import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import RatingsDataset
from two_tower import TwoTowerModel

full_df = pd.read_csv("data/amazon_reviews.csv")
full_df['user_idx'] = full_df['user_id'].astype('category').cat.codes
full_df['item_idx'] = full_df['item_id'].astype('category').cat.codes

n_users = full_df['user_idx'].max() + 1
n_items = full_df['item_idx'].max() + 1

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_ds = RatingsDataset(train_df)
test_ds = RatingsDataset(test_df)
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1024)

model = TwoTowerModel(n_users, n_items, embedding_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for users, items, ratings in train_loader:
        optimizer.zero_grad()
        preds = model(users, items)
        loss = loss_fn(preds, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")

model.eval()
total_test_loss = 0
with torch.no_grad():
    for users, items, ratings in test_loader:
        preds = model(users, items)
        loss = loss_fn(preds, ratings)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss (MSE): {avg_test_loss:.4f}")
print(f"Test RMSE: {avg_test_loss ** 0.5:.4f}")

torch.save(model.state_dict(), "model/two_tower.pt")
print("Model saved.")