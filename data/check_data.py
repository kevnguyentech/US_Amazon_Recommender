# data/inspect.py
import pandas as pd

df = pd.read_csv("data/amazon_reviews.csv")
print(df.shape)
print(df.isnull().sum())
print(df['rating'].describe())