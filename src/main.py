from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import pandas as pd

from combine_train_test import merge_csv
from preprocessing import preprocessing
from GAMO import train_gamo_pipeline
from FCM import fcm_downsample_majority
from GAMOgen import generate_all_classes
from mGWO import mGWO
from SIDS import SIDS_pipeline
from AIDS import anomaly_detection_pipeline_binary, only_predict
from gen_report import generate_final_reports
import os

if __name__ == "__main__":
    
    # merge_csv("./origin_dataset/UNSW_NB15_training-set.csv", "./origin_dataset/UNSW_NB15_testing-set.csv", output="./inter_data/combined.csv")

    # Standardscaling for numeric column, MinMaxScaling for categorical column

    # df = pd.read_csv("./inter_data/combined.csv")

    # df_majority, df_minority = preprocessing(df)
    
    # GAMOpreprocessing(df_minority)

    target_column = "attack_cat"
    # X = df_minority.drop(columns=[target_column])
    # print("X.shape:", X.shape)
    # Y = df_minority[target_column]

    # df_minority_train, df_minority_test = train_test_split(df_minority, test_size=0.2, random_state=42)

    # mlp, dis, gen = train_gamo_pipeline(df_train=df_minority_train, df_test=df_minority_test)

    # num_gen_dict = {
    #     0: 33263,
    #     1: 32260,
    #     2: 31647,
    #     3: 31393,
    #     4: 22902,
    #     5: 21129,
    #     6: 15209,
    #     7: 0
    # }

    # label_mapping = {
    #     0: "Worms",
    #     1: "Shellcode",
    #     2: "Backdoor",
    #     3: "Analysis",
    #     4: "Reconnaissance",
    #     5: "DoS",
    #     6: "Fuzzers",
    #     7: "Exploits"
    # }
    
    # feature_names = df_minority.drop(columns=["attack_cat"]).columns.tolist()

    # # 批量生成
    # df_minority_balanced = generate_all_classes(
    #     gen=gen,
    #     num_gen_dict=num_gen_dict,
    #     latDim=32,
    #     feature_dim=42,
    #     feature_names=feature_names,
    #     c=8,
    #     label_mapping=label_mapping,
    #     original_df=df_minority,
    #     save_path="./inter_data/balanced_minority.csv"        
    # )

    # target_dict = {
    #     "Normal": 80912,
    #     "Generic": 31204
    # }

    # # FCM

    # df_majority = pd.read_csv("./inter_data/majority_class.csv")

    # df_minority_balanced = pd.read_csv("./inter_data/balanced_minority.csv")

    # target_column = "attack_cat"

    # target_dict = {
    #     "Normal": 80912,
    #     "Generic": 31204
    # }

    # df_majority_after_FCM = fcm_downsample_majority(
    #     df=df_majority,
    #     target_column=target_column,
    #     target_dict=target_dict,
    #     n_clusters=166,
    #     m=1.013,
    #     random_state=42
    # )

    # df_majority_after_FCM.to_csv("./inter_data/FCM_0.csv")

    # # 移除 ID 欄位（不分大小寫）
    # for df in [df_majority_after_FCM, df_minority_balanced]:
    #     for col in df.columns:
    #         if col.lower() == "id":
    #             df.drop(columns=[col], inplace=True)

    # # 合併
    # df_concated = pd.concat([df_majority_after_FCM, df_minority_balanced], axis=0, ignore_index=True)
    # df_concated.to_csv("./inter_data/concated_0.csv")

    # # 生成一個假資料集
    # target_column = "attack_cat"
    # X = df_concated.drop(columns=target_column)
    # y = df_concated[target_column]

    # best_feature = mGWO(X, y, pop_size=20, max_iter=86)

    # selected_idx = np.where(best_feature == 1)[0]

    # # 取得選取的特徵欄位名稱
    # selected_columns = df_concated.drop(columns=target_column).columns[selected_idx]

    # # 將 df_concated 篩選成只包含選取的特徵 + target
    # df_selected = df_concated[selected_columns.tolist() + [target_column]]

    # df_selected.to_csv("./inter_data/selected_feat_0.csv", index=False)

    # # 查看結果
    # print("Selected columns + target:")
    # print(df_selected.head())

    df_selected = pd.read_csv("./inter_data/selected_feat_0.csv")

    # df_concated = pd.concat([df_majority, df_minority], axis=0, ignore_index=True)

    df_train, df_test = train_test_split(df_selected, test_size=0.2, random_state=42)

    # df_SIDS_report, SIDS_pred, df_norm_pred, df_pred_non_norm = SIDS_pipeline(df_train=df_train, df_test=df_test, n_trials=10)

    # report = anomaly_detection_pipeline_binary(df_selected, episodes=5)

    # report, AIDS_pred = only_predict(df=df_norm_pred)

    # print ("SIDS_pred = ", SIDS_pred)
    # print ("AIDS_pred = ", AIDS_pred)

    direcPath=os.path.join("./results", "prediction")

    # if not os.path.exists(direcPath):
    #     os.makedirs(direcPath)

    # resSave=os.path.join(direcPath, 'SIDS')
    # np.savez(resSave, SIDS_pred=SIDS_pred)
    # recordSave=os.path.join(direcPath, 'AIDS')
    # np.savez(recordSave, AIDS_pred=AIDS_pred)

    # df_norm_pred.to_csv("./results/prediction/normal.csv")
    # df_pred_non_norm.to_csv("./results/prediction/non_normal.csv")

    data = np.load('./results/prediction/SIDS.npz')
    SIDS_pred = data['SIDS_pred']

    data = np.load('./results/prediction/AIDS.npz')
    AIDS_pred = data['AIDS_pred']

    df_norm_pred = pd.read_csv('./results/prediction/normal.csv')
    df_pred_non_norm = pd.read_csv('./results/prediction/non_normal.csv')


    final_pred_multiclass, final_pred_binary = generate_final_reports(SIDS_pred, df_test, df_norm_pred, df_pred_non_norm, AIDS_pred)
