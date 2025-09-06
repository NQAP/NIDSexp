import pandas as pd

# 假設原始資料集
df = pd.read_csv("./extra_dataset/combined_after_preprocessing.csv")

# 你要的欄位索引 (0-based)
cols = [ 0,2,3,5,8,10,11,12,15,16,17,18,19,22,23,24,25,27,30,31,36,37,39, 42]

# 用 iloc 選取
df_selected = df.iloc[:, cols]

# 存成 CSV
df_selected.to_csv("./extra_dataset/selected_features_with_GAMO.csv", index=False)

print("已經把指定的 features 存到 selected_features.csv")
