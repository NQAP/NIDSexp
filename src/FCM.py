import pandas as pd
import numpy as np
import skfuzzy as fuzz
import optuna

# -----------------------------
# 1. 載入資料（假設多數類資料）
# -----------------------------
# 假設 CSV 已經只包含多數類資料
df_majority = pd.read_csv("./extra_dataset/encoded_majority_class.csv")
target_column = "attack_cat"
X = df_majority.drop(columns=target_column)
X = X.values.T  # FCM 需要 shape = (features, samples)

# -----------------------------
# 2. 定義 Fuzzy C-Means 聚類函數
# -----------------------------
def fcm_cluster(X, n_clusters, m):
    """
    X: shape = (features, samples)
    n_clusters: 聚類數量
    m: 模糊係數
    """
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X, c=n_clusters, m=m, error=0.005, maxiter=1025, init=None
    )
    cluster_labels = np.argmax(u, axis=0)  # 每筆資料最可能的叢集
    return cluster_labels, fpc  # fpc = fuzzy partition coefficient

# -----------------------------
# 3. 使用 OPTUNA 優化 FCM 超參數
# -----------------------------
# def objective(trial):
#     n_clusters = trial.suggest_int("n_clusters", 2, 10)
#     m = trial.suggest_float("m", 1.5, 3.0)
#     _, fpc = fcm_cluster(X, n_clusters, m)
#     # 目標是最大化 FPC (越接近 1 越好)
#     return -fpc  # Optuna 預設是 minimization

# study = optuna.create_study()
# study.optimize(objective, n_trials=30)

# best_params = study.best_params
# print("Best Params:", best_params)

# -----------------------------
# 4. 使用最佳參數進行 FCM 聚類
# -----------------------------
cluster_labels, _ = fcm_cluster(X, 166, 1.013)

# -----------------------------
# 5. 將 Cluster Label 合併回原始資料
# -----------------------------
df_majority["ClusterLabel"] = cluster_labels

# -----------------------------
# 6. 分組操作（可依 ClusterLabel 做後續處理）
# -----------------------------
grouped = df_majority.groupby("ClusterLabel")
for cluster_id, group in grouped:
    print(f"Cluster {cluster_id} has {len(group)} samples")
    # 這裡可以進行生成新樣本、平衡資料等操作

# -----------------------------
# 7. 移除 ClusterLabel 欄位（如果不需要）
# -----------------------------
df_final = df_majority.drop(columns=["ClusterLabel"])
print(df_final.head())
