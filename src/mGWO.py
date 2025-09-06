import numpy as np
import pandas as pd
import random
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import json

# -----------------------------
# 計算 fitness function
# -----------------------------
def compute_fitness(features, X, y, alpha=0.5):
    # 選取被啟用的特徵
    count = 0
    idx = np.where(features == 1)[0]
    if len(idx) == 0:  # 沒有選特徵時，給一個很差的分數
        return -1

    X_sub = X.iloc[:, idx]

    # 訓練/測試集切分
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y, test_size=0.2, random_state=42
    )

    # 使用 XGBoost 做分類
    model = XGBClassifier(eval_metric="mlogloss", device = "cuda")  # 重要：使用 GPU
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 計算混淆矩陣
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    denom = tp + tn + fp + fn
    accuracy = (tp + tn) / denom if denom != 0 else 0
    error_rate = (fp + fn) / denom if denom != 0 else 1  # 錯誤率可視需求設 1
    FAR = fp / (fp + tn + 1e-6)  # 避免除以 0

    total_features = X.shape[1]

    for i in range(len(features)):
        if features[i] == 1:
            count = count + 1
    
    feature_subset = count
    subset_ratio = feature_subset / total_features

    fitness = accuracy - (alpha * (error_rate + FAR) + (1 - alpha) * subset_ratio)

    return fitness


def compute_wolf_correlation(wolf_vector, data):
    """
    計算每隻狼的相關性，固定以 F1 為基準。
    
    參數:
    - wolf_vector: 長度 = 特徵數量的二進位向量，1 表示選中該特徵
    - data: shape = (樣本數, 特徵數)，原始資料集
    
    回傳:
    - corr_value: 該隻狼的相關性值
    """
    # 找出選中的特徵索引
    selected_idx = np.where(wolf_vector == 1)[0]
    
    # 如果 F1 沒被選中，也把它加入計算
    if 0 not in selected_idx:
        selected_idx = np.insert(selected_idx, 0, 0)

    if isinstance(data, pd.DataFrame):
        data = data.values  # 直接轉成 NumPy array
    
    # 取出選中的特徵資料
    selected_data = data[:, selected_idx]

    cov_list = []
    std_list = []
    
    # 以 F1 為基準，計算 F1 與其他選中特徵的相關性
    F1 = selected_data[:, 0]
    for i in range(1, selected_data.shape[1]):
        Fi = selected_data[:, i]
        F1 = np.array(F1, dtype=float)  # 或確保原本就是 array
        Fi = np.array(Fi, dtype=float)
        cov = np.cov(F1, Fi, bias=True)[0, 1]  # 協方差
        std_Fi = np.std(Fi)
        cov_list.append(cov)
        std_list.append(std_Fi)

    corr = np.sum(cov_list) / (data.shape[1] * np.sum(std_list))
    
    return corr

def modified_init_of_pop(train_data, pop_size, dim, iter=50):
    improved_population = []
    wolf_corr = []
    p = 25 / dim
    wolves_pop = np.random.choice([0, 1], size=(pop_size, dim), p=[1-p, p])
    for i in range(len(wolves_pop)):
        corr = compute_wolf_correlation(wolves_pop[i], train_data)
        wolf_corr.append(abs(corr))
    for i in range(len(wolf_corr)):
        if wolf_corr[i] < 0.6:
            improved_population.append(wolves_pop[i])
            print(f"feature subset within a wolf[{i}] is choosed!")
        else:
            print(f"feature subset within a wolf[{i}] is highly correlated!")
    return improved_population

# -----------------------------
# Modified GWO with Improved Initialization
# -----------------------------
def mGWO(X, y, pop_size=10, max_iter=5):
    num_features = X.shape[1]
    # -----------------------------
    # Initialize Population
    # -----------------------------

    # -----------------------------
    # Improved Initialization: Feature Correlation Filtering
    # -----------------------------
    improved_population = modified_init_of_pop(train_data=X, pop_size=pop_size, dim=num_features, iter=max_iter)

    # 若篩選後 population 過少，補回隨機 wolf
    pop_size = len(improved_population)

    population = np.array(improved_population)

    # 初始 fitness 計算
    fitness_scores = np.array([compute_fitness(population[i], X, y) for i in tqdm(range(len(population)))])

    # -----------------------------
    # GWO Iteration
    # -----------------------------
    
    # 排序 (大到小)
    sorted_idx = np.argsort(-fitness_scores)
    population = population[sorted_idx]
    fitness_scores = fitness_scores[sorted_idx]

    # 定義 alpha, beta, delta
    alpha, beta, delta = population[0], population[1], population[2]

    # 更新 a
    

    new_population = []
    for i in tqdm(range(max_iter)):
        a = 2 - 2 * (i / max_iter)

        for j in tqdm(range(pop_size)):
            wolf = population[j]
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha - wolf)
            X1 = alpha[j] - A1 * D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta - wolf)
            X2 = beta[j] - A2 * D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta - wolf)
            X3 = delta[j] - A3 * D_delta

            X_new = (X1 + X2 + X3) / 3
            X_norm = X_new
            for k in range(len(X_new)):
                X_norm[k] = 1 if X_new[k] > 0.5 else 0
            fitness_new = compute_fitness(X_norm, X, y)
            if fitness_new > fitness_scores[j]:
                population[j] = X_norm
                fitness_scores[j] = fitness_new

            new_population.append(population[j])

        population = np.array(new_population)
        sorted_idx = np.argsort(-fitness_scores)
        population = population[sorted_idx]
        fitness_scores = fitness_scores[sorted_idx]

        # 定義 alpha, beta, delta
        alpha, beta, delta = population[0], population[1], population[2]

    # 找到最佳解
    opt_subset_features = alpha
    return opt_subset_features


# -----------------------------
# 測試範例 (隨機資料)
# -----------------------------
if __name__ == "__main__":

    np.random.seed(40)

    df = pd.read_csv("./extra_dataset/combined_0.csv")
    # 生成一個假資料集
    target_column = "attack_cat"
    X = df.drop(columns=target_column)
    y = df[target_column]
    X.info()
    y.info()

    le = LabelEncoder()
    y = le.fit_transform(y)
    encoding_maps = {cls: int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
    print(f"\n欄位 {target_column} 的對應關係： {encoding_maps}")
    with open("./extra_dataset/GWO_label_encodings.json", "w", encoding="utf-8") as f:
        json.dump(encoding_maps, f, ensure_ascii=False, indent=4)

    best_feature = mGWO(X, y, pop_size=20, max_iter=86)

    selected_idx = np.where(best_feature == 1)[0]

    print("最佳特徵子集:", selected_idx)
