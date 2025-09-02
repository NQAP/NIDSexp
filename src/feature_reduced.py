import pandas as pd

# 假設原始資料集
df = pd.read_csv("./extra_dataset/combined_oversampling.csv")

# 你要的欄位索引 (0-based)
cols = [0, 3, 4, 6, 8, 13, 16, 20, 24, 26, 27, 31, 32, 37, 39, 41, 42]

# 用 iloc 選取
df_selected = df.iloc[:, cols]

# 存成 CSV
df_selected.to_csv("./extra_dataset/selected_features.csv", index=False)

print("已經把指定的 features 存到 selected_features.csv")
