# data/prepare.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/amazon_reviews.csv")

df['user_idx'] = df['user_id'].astype('category').cat.codes
df['item_idx'] = df['item_id'].astype('category').cat.codes

n_users = df['user_idx'].nunique()
n_items = df['item_idx'].nunique()
print(f"Users: {n_users}, Items: {n_items}")

# keep only users and items with at least 5 ratings
user_counts = df['user_id'].value_counts()
item_counts = df['item_id'].value_counts()

df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]
df = df[df['item_id'].isin(item_counts[item_counts >= 5].index)]

print(f"After filtering: {df.shape}")

train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print(f"Train: {len(train)}, Test: {len(test)}")