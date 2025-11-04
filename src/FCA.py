from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# from combine_train_test import merge_csv
from preprocessing import preprocessing, major_minor_sep
from GAMO import train_gamo_pipeline
from FCM import fcm_downsample_majority
from GAMOgen import generate_all_classes
from mGWO import mGWO
from SIDS import SIDS_pipeline, test_SIDS_model
from AIDS import anomaly_detection_pipeline_binary, AIDS_predict
from gen_report import generate_final_reports
import os

if __name__ == "__main__":
    
    target_column = "Label"
    df_selected = pd.read_parquet("./processed_fca_real/train_sampled.parquet")

    # SIDS/AIDS Training
    # voting_model = SIDS_pipeline(df_train=df_selected, target_column=target_column, encoding_mapping=None)
    agent = anomaly_detection_pipeline_binary(df=df_selected, target_column=target_column)
    attacks = ['xss', 'nikto', 'hydra', 'nmap', 'sqlmap']
    types = ['FCA', 'noFCA']
    xss = []
    nikto = []
    hydra = []
    nmap = []
    sqlmap = []
    for type in types:
        for attack in attacks:
            df_test = pd.read_parquet(f"./processed_fca_real/test_preprocessed_FCA_real_{attack}_{type}.parquet")
            print(df_test.value_counts())
            SIDS_pred = test_SIDS_model(model=None, df_test=df_test, target_column=target_column, encoding_mapping=None, save_path=f"./FCA_1/results/SIDS_{attack}_{type}_report.csv")
            AIDS_pred = AIDS_predict(df=df_test, target_column=target_column, save_path=f"./FCA_1/results/AIDS_{attack}_{type}_report.csv")
            f1_binary = generate_final_reports(encoding_mapping=None, target_column=target_column, df_test=df_test, SIDS_pred=SIDS_pred, AIDS_pred=AIDS_pred, saving_path=f"{attack}_{type}")
            if attack == 'xss':
                xss.append(f1_binary)
            elif attack == 'nikto':
                nikto.append(f1_binary)
            elif attack == 'hydra':
                hydra.append(f1_binary)
            elif attack == 'nmap':
                nmap.append(f1_binary)
            elif attack == 'sqlmap':
                sqlmap.append(f1_binary)


    # 各攻擊類別顏色
    colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00', '#CC79A7']

    # 圖形設定
    x = np.arange(len(types))
    width = 0.1

    fig, ax = plt.subplots()

    bars1 = ax.bar(x- width * 2, xss, width, label='xss', color=colors[0], edgecolor='k')
    bars2 = ax.bar(x- width + 0.01, nikto, width, label='nikto', color=colors[1], edgecolor='k')
    bars3 = ax.bar(x + 0.02, hydra, width, label='hydra', color=colors[2], edgecolor='k')
    bars4 = ax.bar(x + width + 0.03, nmap, width, label='nmap', color=colors[3], edgecolor='k')
    bars5 = ax.bar(x + width * 2 + 0.04, sqlmap, width, label='sqlmap', color=colors[4], edgecolor='k')

    # 顯示數值在長條上方
    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

    # 中間分隔虛線
    ax.axvline(len(types)/2 - 0.5, color='gray', linestyle='--', linewidth=1)

    # 標籤與樣式
    ax.set_ylabel('F1-score (0–1)')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_title('ARLHNDS F1-score: General vs FCA')

    ax.legend(title='Attack')

    plt.tight_layout()
    plt.show()