import pandas as pd

df = pd.read_csv("./extra_dataset/combined_oversampling.csv")
df.info()
print(df["attack_cat"].value_counts())