import pandas as pd
import numpy as np
import skfuzzy as fuzz

# -----------------------------
# 1. 載入資料
# -----------------------------
df_majority = pd.read_csv("./extra_dataset/majority_class.csv")
target_column = "attack_cat"

# FCM 需要 shape = (features, samples)
X = df_majority.drop(columns=target_column).values.T  

# -----------------------------
# 2. FCM 聚類函數
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
cluster_labels, _ = fcm_cluster(X, n_clusters=166, m=1.013)
df_majority["ClusterLabel"] = cluster_labels

# -----------------------------
# 4. 設定每個 target 類別的總數
# -----------------------------
target_dict = {
    "Normal": 80912,
    "Generic": 31204
}

# -----------------------------
# 5. 根據 cluster 大小分類
# -----------------------------
def classify_clusters(cluster_sizes):
    large = []
    small = []
    tiny = []
    for c, size in cluster_sizes.items():
        if size > 20000:
            large.append(c)
        elif size >= 50:
            small.append(c)
        else:
            tiny.append(c)
    return large, small, tiny

# -----------------------------
# 6. 在每個 target 類別內縮減
# -----------------------------
df_downsampled = []

for target_value, target_count in target_dict.items():
    df_sub = df_majority[df_majority[target_column] == target_value]
    total_sub = len(df_sub)
    cluster_sizes = df_sub.groupby("ClusterLabel").size()
    
    # 分類 cluster
    large_clusters, small_clusters, tiny_clusters = classify_clusters(cluster_sizes)
    
    # 計算大 cluster 縮減比例 r
    large_total = cluster_sizes[large_clusters].sum()
    small_total = cluster_sizes[small_clusters].sum()
    r = (target_count - small_total) / large_total
    
    # 逐 cluster 抽樣
    for cluster_id, group in df_sub.groupby("ClusterLabel"):
        if cluster_id in tiny_clusters:
            continue  # 刪掉極小 cluster
        elif cluster_id in large_clusters:
            n_keep = int(np.floor(len(group) * r))
        else:  # 中小 cluster
            n_keep = len(group)
        
        if n_keep > 0:
            df_downsampled.append(group.sample(n=n_keep, random_state=42))

# -----------------------------
# 7. 合併並清理
# -----------------------------
df_final = pd.concat(df_downsampled).reset_index(drop=True)
df_final = df_final.drop(columns=["ClusterLabel"])
df_final.to_csv("./extra_dataset/major_after_reduced_final.csv", index=False)

# -----------------------------
# 8. 檢視結果
# -----------------------------
print(df_final.head())
print("Original size:", len(df_majority), "Downsampled size:", len(df_final))
print(df_final[target_column].value_counts())
