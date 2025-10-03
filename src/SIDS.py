import optuna
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt

# =============== 超參數最佳化函數 (以 RandomForest 為例) ===============
# def objective_rf(trial, X, y):
#     n_estimators = trial.suggest_int("n_estimators", 50, 300)
#     max_depth = trial.suggest_int("max_depth", 3, 20)
#     min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
#     clf = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         random_state=42,
#         n_jobs=-1
#     )
#     score = cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()
#     return score

# 針對每個模型進行 optuna 調參
# def tune_model(model_name, X, y, n_trials=20):
#     if model_name == "RF":
#         study = optuna.create_study(direction="maximize")
#         study.optimize(lambda trial: objective_rf(trial, X, y), n_trials=n_trials)
#         return study.best_params
#     elif model_name == "DT":
#         study = optuna.create_study(direction="maximize")
#         def objective(trial):
#             max_depth = trial.suggest_int("max_depth", 3, 20)
#             min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
#             clf = DecisionTreeClassifier(
#                 max_depth=max_depth,
#                 min_samples_split=min_samples_split,
#                 random_state=42
#             )
#             return cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()
#         study.optimize(objective, n_trials=n_trials)
#         return study.best_params
#     # 其他模型可依需求加上 (這裡先簡化示範)
#     return {}

# ---------- 自訂投票函數 ----------
def voting_predict(estimators, X, voting="hard"):
    preds_list = []
    for est in estimators:
        pred = est.predict(X)
        if pred.ndim > 1:  # 如果是 2D array，例如 CatBoost
            pred = np.ravel(pred)  # 壓平成 1D
        preds_list.append(pred)
    preds = np.asarray(preds_list)

    if voting == "hard":
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
    elif voting == "soft":
        probs_list = []
        for est in estimators:
            if hasattr(est, "predict_proba"):
                probs_list.append(est.predict_proba(X))
            else:  # CatBoost
                probs_list.append(est.predict(X, prediction_type="Probability"))
        probs = np.mean(probs_list, axis=0)
        return np.argmax(probs, axis=1)
    else:
        raise ValueError("voting must be 'hard' or 'soft'")

# =============== 主程式 (SIDS) ===============


class CustomVotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators
    
    def predict(self, X, voting="hard"):
        preds_list = []
        for est in self.estimators:
            pred = est.predict(X)
            if pred.ndim > 1:
                pred = np.ravel(pred)
            preds_list.append(pred)
        preds = np.asarray(preds_list)

        if voting == "hard":
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
        elif voting == "soft":
            probs_list = []
            for est in self.estimators:
                if hasattr(est, "predict_proba"):
                    probs_list.append(est.predict_proba(X))
                else:  # CatBoost 需要特殊處理
                    probs_list.append(est.predict(X, prediction_type="Probability"))
            probs = np.mean(probs_list, axis=0)
            return np.argmax(probs, axis=1)
        else:
            raise ValueError("voting must be 'hard' or 'soft'")

# ================= 主程式 =================
def SIDS_pipeline(df_train, target_column, encoding_mapping, n_trials=20, save_model=True):

    df_train[target_column] = df_train[target_column].map(encoding_mapping)
    X_train, y_train = df_train.drop(target_column, axis=1), df_train[target_column]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 建立與訓練模型
    dt = DecisionTreeClassifier(
        splitter='random',
        criterion='gini',
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=14,
        random_state=42
    ).fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=128,
        max_depth=16,
        min_samples_split=24,
        min_samples_leaf=10,
        random_state=42
    ).fit(X_train, y_train)

    et = ExtraTreesClassifier(
        n_estimators=950,
        criterion='gini',
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=False,
        random_state=42
    ).fit(X_train, y_train)

    ab = AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.579,
        algorithm='SAMME',
        random_state=42
    ).fit(X_train, y_train)

    lgbm = LGBMClassifier(
        learning_rate=0.007,
        n_estimators=287,
        num_leaves=35,
        max_depth=5,
        random_state=42
    ).fit(X_train, y_train)

    cb = CatBoostClassifier(
        min_data_in_leaf=47,
        learning_rate=0.0069,
        iterations=284,
        depth=8,
        l2_leaf_reg=0.80908,
        verbose=0,
        random_state=42
    ).fit(X_train, y_train)

    xgb = XGBClassifier(
        booster='dart',
        reg_lambda=0.004,
        reg_alpha=2.6e-04,
        subsample=0.367,
        colsample_bytree=0.923,
        n_estimators=16,
        early_stopping_rounds=12,
        max_depth=7,
        min_child_weight=10,
        eta=0.1163,
        gamma=4.57e-05,
        grow_policy='lossguide',
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # ---------------- 包裝投票模型 ----------------
    estimators = [dt, rf, et, ab, lgbm, cb, xgb]
    voting_model = CustomVotingClassifier(estimators)
    print (y_train.value_counts())

    y_pred = voting_model.predict(X_val)

    report = classification_report(y_val, y_pred, output_dict=True)
    if encoding_mapping is not None:
        inv_mapping = {v: k for k, v in encoding_mapping.items()}
        report_converted = {}
        for key, value in report.items():
            try:
                int_key = int(key)
                new_key = inv_mapping[int_key]
            except:
                new_key = key
            report_converted[new_key] = value
    else:
        report_converted = report

    # --- Step 5: 存成 CSV ---
    rows = []
    for cls, metrics in report_converted.items():
        if isinstance(metrics, dict):
            rows.append({
                "class": cls,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
                "support": int(metrics["support"])
            })
        else:
            rows.append({
                "class": cls,
                "precision": "",
                "recall": "",
                "f1-score": "",
                "support": metrics
            })

    df_report = pd.DataFrame(rows)
    save_path = "./FCA/results/SIDS_val.csv"
    df_report.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"📊 測試報告已儲存於 {save_path}")
    print(df_report)
    

    # ---------------- 存檔 ----------------
    if save_model:
        joblib.dump(voting_model, "./model/FCA/sids_voting_model.pkl")
        print("✅ 投票模型已儲存為 ./model/FCA/sids_voting_model.pkl")

    return voting_model


    # ---------- 報告 ----------
def test_SIDS_model(model, df_test, target_column, encoding_mapping=None, save_path="./FCA/results/SIDS_report.csv"):
    """
    model: 訓練好的模型
    df_test: 測試資料 DataFrame
    target_column: 標籤欄位名稱
    encoding_mapping: 如果 label 需要 encode
    normal_label: 被判定為 Normal 的 label 數字
    save_path: CSV 報告存放路徑
    """
    # --- Step 1: Label encode (需跟訓練時一致) ---
    if encoding_mapping is not None:
        df_test[target_column] = df_test[target_column].map(encoding_mapping)
        normal_label = encoding_mapping.get("Normal")

    if model is None:
        model = joblib.load("./model/FCA/sids_voting_model.pkl")
    
    X_test, y_test = df_test.drop(columns=[target_column]), df_test[target_column]
    print(y_test.value_counts())
    # --- Step 2: 預測 ---
    y_pred = model.predict(X_test)
    print(y_pred)
    # --- Step 3: 生成分類報告 ---
    report = classification_report(y_test, y_pred, output_dict=True)
    print(confusion_matrix(y_test, y_pred))

    # --- Step 4: 還原 label (如果有 mapping 的話) ---
    if encoding_mapping is not None:
        inv_mapping = {v: k for k, v in encoding_mapping.items()}
        report_converted = {}
        for key, value in report.items():
            try:
                int_key = int(key)
                new_key = inv_mapping[int_key]
            except:
                new_key = key
            report_converted[new_key] = value
    else:
        report_converted = report

    # --- Step 5: 存成 CSV ---
    rows = []
    for cls, metrics in report_converted.items():
        if isinstance(metrics, dict):
            rows.append({
                "class": cls,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
                "support": int(metrics["support"])
            })
        else:
            rows.append({
                "class": cls,
                "precision": "",
                "recall": "",
                "f1-score": "",
                "support": metrics
            })

    df_report = pd.DataFrame(rows)
    df_report.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"📊 測試報告已儲存於 {save_path}")
    print(df_report)

    # --- Step 6: 取得被判定為 Normal 的資料 ---
    normal_idx = np.where(y_pred == normal_label)[0]
    non_normal_idx = np.where(y_pred != normal_label)[0]

    df_pred_normal = df_test.iloc[normal_idx].copy()
    df_pred_non_normal = df_test.iloc[non_normal_idx].copy()

    print(f"✅ 被判定為 Normal 的資料數量: {len(df_pred_normal)}")
    print(f"⚠️ 被判定為 非 Normal 的資料數量: {len(df_pred_non_normal)}")

    return y_pred, df_pred_normal, df_pred_non_normal


# =============== 測試用範例 (模擬資料集) ===============
if __name__ == "__main__":

    df = pd.read_csv("./extra_dataset/combined_2.csv")

    target_column = "attack_cat"
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    encoding_maps = {cls: int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
    print(f"\n欄位 {target_column} 的對應關係： {encoding_maps}")
    # 儲存對應關係到 JSON 檔
    with open("./extra_dataset/SIDS_label_encodings.json", "w", encoding="utf-8") as f:
        json.dump(encoding_maps, f, ensure_ascii=False, indent=4)

    df_train, df_test = train_test_split(df, test_size=0.2)

    df_train.info()
    df_test.info()

    print(SIDS_pipeline(df_train, df_test, n_trials=10))
