import pandas as pd
import numpy as np
import skfuzzy as fuzz

def fcm_downsample_majority(
    df,
    target_column,
    target_dict,
    n_clusters=166,
    m=1.013,
    random_state=42,
    keep_low_membership_ratio=0.1,
):
    """
    FCM 聚類後，對每個 class 下採樣，保證總數量等於 target_dict，
    並可保留部分低隸屬樣本。

    參數:
        df (pd.DataFrame): 輸入資料
        target_column (str): 標籤欄位
        target_dict (dict): {class_name: 保留數量}
        n_clusters (int): FCM 聚類數
        m (float): 模糊係數
        random_state (int): 隨機種子
        keep_low_membership_ratio (float): 保留低隸屬樣本比例

    回傳:
        pd.DataFrame: 下採樣後結果
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 1. FCM 聚類
    X = df.drop(columns=target_column).values.T
    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        X, c=n_clusters, m=m, error=0.005, maxiter=1025, init=None
    )
    cluster_labels = np.argmax(u, axis=0)
    df = df.copy()
    df["ClusterLabel"] = cluster_labels

    df_downsampled = []

    # 2. 對每個 class 處理
    for class_name, target_count in target_dict.items():
        df_sub = df[df[target_column] == class_name]
        total_sub = len(df_sub)
        cluster_sizes = df_sub.groupby("ClusterLabel").size()

        # 每個 cluster 高隸屬分配數量
        num_per_cluster = {
            cluster_id: int(np.round(size / total_sub * target_count * (1 - keep_low_membership_ratio)))
            for cluster_id, size in cluster_sizes.items()
        }

        # 抽樣
        selected_indices = []
        for cluster_id, group in df_sub.groupby("ClusterLabel"):
            idxs = group.index
            membership_scores = u[cluster_id, idxs]

            # 高隸屬樣本
            n_keep_high = num_per_cluster.get(cluster_id, 0)
            if n_keep_high >= len(group):
                selected_indices.extend(idxs.tolist())
            elif n_keep_high > 0:
                top_idx = idxs[np.argsort(-membership_scores)[:n_keep_high]]
                selected_indices.extend(top_idx.tolist())

            # 低隸屬樣本
            n_keep_low = int(len(group) * keep_low_membership_ratio)
            # 確保總數不超過 target_count
            remaining = target_count - len(selected_indices)
            n_keep_low = min(n_keep_low, remaining)
            if n_keep_low > 0:
                low_idx = idxs[np.argsort(membership_scores)[:n_keep_low]]
                selected_indices.extend(low_idx.tolist())

        # 避免超過 target_count（因四捨五入可能會多）
        if len(selected_indices) > target_count:
            selected_indices = np.random.choice(selected_indices, target_count, replace=False)
        df_downsampled.append(df.loc[selected_indices])

    df_sampled = pd.concat(df_downsampled).reset_index(drop=True)
    df_sampled = df_sampled.drop(columns=["ClusterLabel"])

    print("Original size:", len(df), "Downsampled size:", len(df_sampled))
    print("FPC (fuzzy partition coefficient):", fpc)
    print(df_sampled[target_column].value_counts())

    return df_sampled
