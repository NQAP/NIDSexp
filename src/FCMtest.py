import pandas as pd
import numpy as np
import skfuzzy as fuzz

# -----------------------------
# 1. 載入資料
# -----------------------------
df_majority = pd.read_csv("./extra_dataset/encoded_majority_class.csv")
df_minority = pd.read_csv("./extra_dataset/encoded_minority_class.csv")  # 少數類
target_column = "attack_cat"

X = df_majority.drop(columns=target_column).values.T  # FCM 需要 shape = (features, samples)

# -----------------------------
# 2. Fuzzy C-Means 聚類函數
# -----------------------------
def fcm_cluster(X, n_clusters, m):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X, c=n_clusters, m=m, error=0.005, maxiter=1025, init=None
    )
    cluster_labels = np.argmax(u, axis=0)
    return cluster_labels, fpc

# -----------------------------
# 3. 聚類
# -----------------------------
print ("cluster start!")
cluster_labels, _ = fcm_cluster(X, n_clusters=166, m=1.013)
df_majority["ClusterLabel"] = cluster_labels
print ("cluster done!")


# -----------------------------
# 4. 指定每個 target value 要保留的數量
# -----------------------------
target_dict = {
    "Normal": 80912,
    "Generic": 31204
}

# -----------------------------
# 5. 在每個 target value 內依 cluster 下採樣
# -----------------------------
df_downsampled = []

for target_value, target_count in target_dict.items():
    df_sub = df_majority[df_majority[target_column] == target_value]
    total_sub = len(df_sub)
    cluster_sizes = df_sub.groupby("ClusterLabel").size()
    
    # 計算每個 cluster 的目標數量
    num_target_per_cluster = {
        cluster_id: int(np.round(size / total_sub * target_count))
        for cluster_id, size in cluster_sizes.items()
    }
    
    # cluster 內隨機抽樣
    for cluster_id, group in df_sub.groupby("ClusterLabel"):
        n_keep = num_target_per_cluster[cluster_id]
        if n_keep >= len(group):
            df_downsampled.append(group)
        else:
            df_downsampled.append(group.sample(n=n_keep, random_state=42))

# -----------------------------
# 6. 合併並清理
# -----------------------------
df_final = pd.concat(df_downsampled).reset_index(drop=True)
df_final = df_final.drop(columns=["ClusterLabel"])
df_final.to_csv("./extra_dataset/major_after_FCM.csv", index=False)

# -----------------------------
# 7. 檢視結果
# -----------------------------
print(df_final.head())
print("Original size:", len(df_majority), "Downsampled size:", len(df_final))