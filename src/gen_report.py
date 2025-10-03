import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def generate_final_reports(
        encoding_mapping, 
        target_column, 
        df_test, 
        df_norm_pred, 
        df_pred_non_norm,
        SIDS_pred, 
        AIDS_pred,
        saving_path
        ):
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
    if encoding_mapping is not None:
        inverse_mapping = {v: k for k,v in encoding_mapping.items()}
        df_test_enc = df_test.copy()
        df_test_enc[target_column] = df_test_enc[target_column].map(inverse_mapping)

    aids_pred_map = dict(zip(df_norm_pred.index, AIDS_pred))

    # --------- 建立最終多分類預測 ---------
    final_pred_multiclass_enc = []
    for idx, pred_sids in zip(df_test.index, SIDS_pred):
        if idx in aids_pred_map:
            # 使用 AIDS 預測結果
            pred = encoding_mapping["Normal"] if aids_pred_map[idx] == 0 else -1
            final_pred_multiclass_enc.append(pred)
        else:
            final_pred_multiclass_enc.append(pred_sids)

    # 將數字 Label 轉回文字
    final_pred_multiclass = [inverse_mapping[x] if x in inverse_mapping else "Attack_from_Normal" for x in final_pred_multiclass_enc]

    y_true_multiclass = df_test[target_column].tolist()

    print(y_true_multiclass[0], final_pred_multiclass[0])

    # --------- 多分類報告 ---------
    report_multiclass_dict = classification_report(
        y_true_multiclass, final_pred_multiclass, output_dict=True
    )
    df_report_multiclass = pd.DataFrame(report_multiclass_dict).transpose()
    df_report_multiclass.to_csv(f"./FCA/results/{saving_path}_multi.csv", encoding="utf-8-sig")
    print(f"多分類報告 CSV 已儲存: ./FCA/results/{saving_path}_multi.csv")

    # --------- 二分類報告 ---------
    y_true_binary = [0 if x == encoding_mapping["Normal"] else 1 for x in df_test_enc[target_column]]
    final_pred_binary = [0 if x == "Normal" else 1 for x in final_pred_multiclass]

    report_binary_dict = classification_report(
        y_true_binary, final_pred_binary, target_names=["Normal","Attack"], output_dict=True
    )
    df_report_binary = pd.DataFrame(report_binary_dict).transpose()
    df_report_binary.to_csv(f"./FCA/results/{saving_path}_binary.csv", encoding="utf-8-sig")
    print(f"二分類報告 CSV 已儲存: ./FCA/results/{saving_path}_binary.csv")

    # --------- Accuracy ---------
    accuracy_multiclass = accuracy_score(y_true_multiclass, final_pred_multiclass)
    accuracy_binary = accuracy_score(y_true_binary, final_pred_binary)

    df_accuracy = pd.DataFrame({
        "Metric": ["Multiclass Accuracy", "Binary Accuracy"],
        "Value": [accuracy_multiclass, accuracy_binary]
    })
    df_accuracy.to_csv(f"./FCA/results/{saving_path}_final_acc.csv", index=False, encoding="utf-8-sig")
    print(f"Accuracy CSV 已儲存: ./FCA/results/{saving_path}_final_acc.csv")

