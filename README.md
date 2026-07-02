## Results & Limitations

**Setup:** 300k Amazon Electronics reviews, filtered to users/items with 5+ ratings (97,646 rows), 55,108 users, 9,809 items, avg 9.95 ratings/item.

**RMSE:** 1.15 (vs 1.24 baseline of predicting the global average rating)

**Precision@5:** 0.858 (vs 0.846 random baseline)

Model beats baseline on RMSE but barely beats random on Precision@5. Ratings skew hard toward 4-5 stars (mean 4.26), so almost any ranking looks "precise" against this metric, including random. The real limiter isn't item sparsity (9.95 ratings/item is workable) but that the model only sees user_id and item_id, no content signal to distinguish *why* someone likes an item.

**Next step:** add item content features (title/description embeddings) so the model has signal beyond just IDs.