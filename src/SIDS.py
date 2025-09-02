import optuna
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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

# =============== 主程式 (SIDS) ===============
def SIDS_pipeline(df_train, df_test, n_trials=20):
    # 分割資料
    X_train, y_train = df_train.drop("attack_cat", axis=1), df_train["attack_cat"]
    X_test, y_test = df_test.drop("attack_cat", axis=1), df_test["attack_cat"]

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # 調參 (示範僅做 DT, RF，其他模型可依需求補齊)
    # best_params_dt = tune_model("DT", X_train, y_train, n_trials=n_trials)
    # best_params_rf = tune_model("RF", X_train, y_train, n_trials=n_trials)

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
    
    rf = RandomForestClassifier(
            n_estimators=128,
            max_depth=16,
            min_samples_split=24,
            min_samples_leaf=10,
            random_state=42,
        )
    
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
    
    ab = AdaBoostClassifier(
            n_estimators=300,
            learning_rate=0.579,
            algorithm='SAMME',
            random_state=42
        )

    lgbm = LGBMClassifier(
            learning_rate=0.007,
            n_estimators=287,
            num_leaves=35,
            max_depth=5,
            random_state=42
        )

    cb = CatBoostClassifier(
            min_data_in_leaf=47,
            learning_rate=0.0069,
            iterations=284,
            depth=8,
            l2_leaf_reg=0.80908,
            verbose=0, 
            random_state=42
        )

    xgb = XGBClassifier(
            booster='dart',
            reg_lambda=0.004,
            reg_alpha=2.6e-04,
            subsample=0.367,
            cosample_bytree=0.923,
            # early_stopping_rounds=12,
            n_estimators=16,
            max_depth=7,
            min_child_weight=10,
            eta=0.1163,
            gamma=4.57e-05,
            grow_policy='lossguide',
            random_state=42, 
            use_label_encoder=False, 
            eval_metric="mlogloss"
        )

    # 多數投票集成
    majority_voting = VotingClassifier(
        estimators=[
            ("DT", dt),
            ("RF", rf),
            ("ET", et),
            ("AB", ab),
            ("LGBM", lgbm),
            ("CB", cb),
            ("XGB", xgb),
        ],
        voting="hard"
    )

    # 訓練
    majority_voting.fit(X_train, y_train)

    # 預測
    predictions = majority_voting.predict(X_test)

    # 報告
    report = classification_report(y_test, predictions)
    return report


# =============== 測試用範例 (模擬資料集) ===============
if __name__ == "__main__":

    df = pd.read_csv("./extra_dataset/selected_features.csv")

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
