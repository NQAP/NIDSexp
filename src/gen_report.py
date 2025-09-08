import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def generate_final_reports(SIDS_pred, df_test, df_norm_pred, df_pred_non_norm, AIDS_pred):
    """
    根據 SIDS 和 AIDS 預測結果生成最終報告
    並輸出 CSV：
      - final_multiclass_report.csv (多分類)
      - final_binary_report.csv (二分類)
      - final_accuracy.csv (Accuracy)
    
    自動處理 Label Encoding 和轉回文字

    參數:
        SIDS_pred : array-like, SIDS 對整個測試集的預測 (數字 Label)
        df_test : 原始測試集 DataFrame (文字 Label)
        df_norm_pred : SIDS 判為 Normal 的 DataFrame
        df_pred_non_norm : SIDS 判為非 Normal 的 DataFrame
        AIDS_pred : array-like, AIDS 對 SIDS Normal 的預測 (0=Normal, 1=Attack)
    回傳:
        final_pred_multiclass : list, 整合 SIDS + AIDS 的最終多分類預測
        final_pred_binary : list, 二分類預測 (0=Normal, 1=Attack)
    """

    # --------- Label Encoding map ---------
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
    num_to_label = {v: k for k, v in label_encoding.items()}

    # 將 df_test 的文字 Label 轉成數字 Label
    df_test_enc = df_test.copy()
    df_test_enc['attack_cat'] = df_test_enc['attack_cat'].map(label_encoding)

    normal_idx = df_norm_pred.index
    non_normal_idx = df_pred_non_norm.index

    aids_pred_map = dict(zip(df_norm_pred.index, AIDS_pred))

    # --------- 建立最終多分類預測 ---------
    final_pred_multiclass_enc = []
    for idx, pred_sids in zip(df_test.index, SIDS_pred):
        if idx in aids_pred_map:
            # 使用 AIDS 預測結果
            pred = label_encoding["Normal"] if aids_pred_map[idx] == 0 else -1
            final_pred_multiclass_enc.append(pred)
        else:
            final_pred_multiclass_enc.append(pred_sids)

    # 將數字 Label 轉回文字
    final_pred_multiclass = [num_to_label[x] if x in num_to_label else "Attack_from_Normal" for x in final_pred_multiclass_enc]

    y_true_multiclass = df_test['attack_cat'].tolist()

    print(y_true_multiclass[0], final_pred_multiclass[0])

    # --------- 多分類報告 ---------
    report_multiclass_dict = classification_report(
        y_true_multiclass, final_pred_multiclass, output_dict=True
    )
    df_report_multiclass = pd.DataFrame(report_multiclass_dict).transpose()
    df_report_multiclass.to_csv("./results/final_multiclass_report.csv", encoding="utf-8-sig")
    print("多分類報告 CSV 已儲存: ./results/final_multiclass_report.csv")

    # --------- 二分類報告 ---------
    y_true_binary = [0 if x == label_encoding["Normal"] else 1 for x in df_test_enc['attack_cat']]
    final_pred_binary = [0 if x == "Normal" else 1 for x in final_pred_multiclass]

    report_binary_dict = classification_report(
        y_true_binary, final_pred_binary, target_names=["Normal","Attack"], output_dict=True
    )
    df_report_binary = pd.DataFrame(report_binary_dict).transpose()
    df_report_binary.to_csv("./results/final_binary_report.csv", encoding="utf-8-sig")
    print("二分類報告 CSV 已儲存: ./results/final_binary_report.csv")

    # --------- Accuracy ---------
    accuracy_multiclass = accuracy_score(y_true_multiclass, final_pred_multiclass)
    accuracy_binary = accuracy_score(y_true_binary, final_pred_binary)

    df_accuracy = pd.DataFrame({
        "Metric": ["Multiclass Accuracy", "Binary Accuracy"],
        "Value": [accuracy_multiclass, accuracy_binary]
    })
    df_accuracy.to_csv("./results/final_accuracy.csv", index=False, encoding="utf-8-sig")
    print("Accuracy CSV 已儲存: ./results/final_accuracy.csv")

    return final_pred_multiclass, final_pred_binary
