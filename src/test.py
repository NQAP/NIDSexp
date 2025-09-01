import pandas as pd

df = pd.read_csv("./extra_dataset/major_after_reduced.csv")
print(df["attack_cat"].value_counts())