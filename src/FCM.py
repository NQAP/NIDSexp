import pandas as pd
import numpy as np
import skfuzzy as fuzz

def fcm_downsample_majority(
    df_majority,
    target_column,
    target_dict,
    n_clusters=166,
    m=1.013,
    output_path=None,
    random_state=42
):
    """
    使用 Fuzzy C-Means 進行聚類，並依據高隸屬度進行下採樣。

    參數:
        df_majority (pd.DataFrame): 輸入 DataFrame
        target_column (str): 分類標籤欄位
        target_dict (dict): {class_name: 保留數量}
        n_clusters (int): 聚類數
        m (float): FCM 模糊係數
        output_path (str): 輸出檔案路徑 (選填)
        random_state (int): 隨機種子 (選填)

    回傳:
        pd.DataFrame: 下採樣後的 DataFrame
    """

    if random_state is not None:
        np.random.seed(random_state)

    # 1. 取特徵矩陣
    X = df_majority.drop(columns=target_column).values.T  # shape = (features, samples)

    # 2. FCM 聚類
    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        X, c=n_clusters, m=m, error=0.005, maxiter=1025, init=None
    )
    cluster_labels = np.argmax(u, axis=0)
    df_majority = df_majority.copy()
    df_majority["ClusterLabel"] = cluster_labels

    # 3. 高隸屬度抽樣函數
    def membership_weighted_sampling(df_sub, cluster_id, n_keep, u):
        group_idx = df_sub.index
        weights = u[cluster_id, group_idx]
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = None
        chosen_idx = np.random.choice(group_idx, size=n_keep, replace=False, p=weights)
        return df_sub.loc[chosen_idx]

    # 4. 在每個 target value 內依 cluster 下採樣
    df_downsampled = []
    for target_value, target_count in target_dict.items():
        df_sub = df_majority[df_majority[target_column] == target_value]
        total_sub = len(df_sub)
        cluster_sizes = df_sub.groupby("ClusterLabel").size()

        num_target_per_cluster = {
            cluster_id: int(np.round(size / total_sub * target_count))
            for cluster_id, size in cluster_sizes.items()
        }

        for cluster_id, group in df_sub.groupby("ClusterLabel"):
            n_keep = num_target_per_cluster[cluster_id]
            if n_keep >= len(group):
                df_downsampled.append(group)
            else:
                df_downsampled.append(membership_weighted_sampling(group, cluster_id, n_keep, u))

    # 5. 合併並清理
    df_final = pd.concat(df_downsampled).reset_index(drop=True)
    df_final = df_final.drop(columns=["ClusterLabel"])

    # 6. 輸出 CSV (若有指定)
    if output_path:
        df_final.to_csv(output_path, index=False)

    print("Original size:", len(df_majority), "Downsampled size:", len(df_final))
    print(df_final[target_column].value_counts())
    print("FPC (fuzzy partition coefficient):", fpc)

    return df_final
