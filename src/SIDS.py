import optuna
import json
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
def SIDS_pipeline(df_train, df_test, n_trials=20):
    # 分割資料

    # 調參 (示範僅做 DT, RF，其他模型可依需求補齊)
    # best_params_dt = tune_model("DT", X_train, y_train, n_trials=n_trials)
    # best_params_rf = tune_model("RF", X_train, y_train, n_trials=n_trials)
    target_column = "attack_cat"
    le = LabelEncoder()
    df_train[target_column] = le.fit_transform(df_train[target_column])
    df_test[target_column] = le.fit_transform(df_test[target_column])
    encoding_maps = {cls: int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
    print(f"\n欄位 {target_column} 的對應關係： {encoding_maps}")
    # 儲存對應關係到 JSON 檔
    with open("./inter_data/SIDS_label_encodings.json", "w", encoding="utf-8") as f:
        json.dump(encoding_maps, f, ensure_ascii=False, indent=4)

    X_train, y_train = df_train.drop("attack_cat", axis=1), df_train["attack_cat"]
    X_test, y_test = df_test.drop("attack_cat", axis=1), df_test["attack_cat"]

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    # 建立模型
    dt = DecisionTreeClassifier(
            splitter='random',
            criterion='gini',
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=14,
            min_weight_fraction_leaf=2.3e-04,
            min_impurity_decrease=7.28e-05,
            max_leaf_nodes=29,
            random_state=42
        )
    dt.fit(X_train, y_train)
    
    rf = RandomForestClassifier(
            n_estimators=128,
            max_depth=16,
            min_samples_split=24,
            min_samples_leaf=10,
            random_state=42,
        )
    rf.fit(X_train, y_train)

    et = ExtraTreesClassifier(
            n_estimators=950,
            criterion='gini',
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=3,
            max_features='sqrt',
            bootstrap=False,
            random_state=42
        )
    et.fit(X_train, y_train)
    
    ab = AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.579,
            algorithm='SAMME',
            random_state=42
        )
    ab.fit(X_train, y_train)

    lgbm = LGBMClassifier(
            learning_rate=0.007,
            n_estimators=287,
            num_leaves=35,
            max_depth=5,
            random_state=42
        )
    lgbm.fit(X_train, y_train)

    cb = CatBoostClassifier(
            min_data_in_leaf=47,
            learning_rate=0.0069,
            iterations=284,
            depth=8,
            l2_leaf_reg=0.80908,
            verbose=0, 
            random_state=42
        )
    cb.fit(X_train, y_train)

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
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    estimators = [dt, rf, et, ab, lgbm, cb, xgb]
    predictions = voting_predict(estimators, X_test, voting="hard")

    # ---------- 報告 ----------
    # 你的 label encoding 對照表
    label_encoding = {
        "Analysis": 0,
        "Backdoor": 1,
        "DoS": 2,
        "Exploits": 3,
        "Fuzzers": 4,
        "Generic": 5,
        "Normal": 6,
        "Reconnaissance": 7,
        "Shellcode": 8,
        "Worms": 9
    }

    # 反轉字典，方便從數字找到文字 label
    num_to_label = {v: k for k, v in label_encoding.items()}

    # 生成 classification report（字典形式）
    report_dict = classification_report(y_test, predictions, output_dict=True)

    # 將 key（數字 label）轉換成文字 label
    report_converted = {}
    for key, value in report_dict.items():
        try:
            int_key = int(key)  # 這裡 key 可能是數字 label
            new_key = num_to_label[int_key]
        except:
            new_key = key  # 其他 key (如 'accuracy', 'macro avg', 'weighted avg')
        report_converted[new_key] = value

    # 儲存 CSV
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
    df_report.to_csv("./results/SIDS_2.csv", index=False, encoding="utf-8-sig")
    print("CSV 已儲存為 ./results/SIDS_2.csv")

    labels = [
        "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
        "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"
    ]

    # 混淆矩陣 (10x10)
    cm = confusion_matrix(y_test, predictions, labels=range(len(labels)))

    print("Confusion Matrix (10 classes):")
    print(cm)

    # 繪圖（Seaborn heatmap）
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("SIDS Confusion Matrix (10 classes)")
    plt.show()

    

    
    # ---------- 取出預測為 Normal 的 row ----------
    normal_label = label_encoding["Normal"]
    df_normal_pred = df_test[predictions == normal_label]

    # 建立反向映射
    inverse_label_encoding = {v: k for k, v in label_encoding.items()}

    df_normal_pred[target_column] = df_normal_pred[target_column].map(inverse_label_encoding)  # 假設 column 名稱是 'label'

    df_pred_non_normal = df_test[predictions != normal_label].copy()
    df_pred_non_normal[target_column] = df_pred_non_normal[target_column].map(inverse_label_encoding)


    print(df_normal_pred.info())
    
    return df_report, predictions, df_normal_pred, df_pred_non_normal


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
