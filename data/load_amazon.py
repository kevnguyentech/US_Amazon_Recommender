from datasets import load_dataset
import pandas as pd

CATEGORY = "raw_review_Electronics"
SAMPLE_SIZE = 300_000

def main():
    print(f"Loading {CATEGORY}...")
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", CATEGORY,
                            trust_remote_code=True, streaming=True)

    rows = []
    for i, row in enumerate(dataset["full"]):
        rows.append({
            "user_id": row["user_id"],
            "item_id": row["parent_asin"],
            "rating": row["rating"],
            "timestamp": row["timestamp"],
        })
        if i >= SAMPLE_SIZE:
            break
        if i % 10_000 == 0:
            print(f"  {i} rows loaded")

    df = pd.DataFrame(rows)
    df.to_csv("data/amazon_reviews.csv", index=False)
    print(f"Saved {len(df)} rows to data/amazon_reviews.csv")

if __name__ == "__main__":
    main()
