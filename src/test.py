from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import pandas as pd
import json

# from combine_train_test import merge_csv
from preprocessing import preprocessing
from GAMO import train_gamo_pipeline
from FCM import fcm_downsample_majority
from GAMOgen import generate_all_classes
from mGWO import mGWO
from SIDS import SIDS_pipeline, test_SIDS_model
from AIDS import anomaly_detection_pipeline_binary, AIDS_predict
from gen_report import generate_final_reports
import os
import re

target_column = "Label"

df = pd.read_csv("./FCA_DATASET/Train/Train_multiple.csv")
df, std_scaler = preprocessing(df, target_column=target_column)

df_concated = pd.read_csv("./FCA/concated_0.csv")
df_concated.info()
df_concated = df_concated.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '_', x))
df_selected = pd.read_csv("./FCA/selected_feat_0.csv")
df_selected["Label"] = df_selected["Label"].apply(lambda x: "Normal" if (x == 0 or x == '0') else x)

df_selected = df_selected.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '_', x))
selected_columns = df_selected.drop(columns=[target_column]).columns


hydra_FCA = pd.read_csv("./FCA_DATASET/Test/hydra_FCA.csv")
hydra_noFCA = pd.read_csv("./FCA_DATASET/Test/hydra_noFCA.csv")

hydra_FCA[target_column] = hydra_FCA[target_column].apply(lambda x: "Normal" if (x == 0 or x == '0') else "Hydra")
hydra_noFCA[target_column] = hydra_noFCA[target_column].apply(lambda x: "Normal" if (x == 0 or x == '0') else "Hydra")

hydra_FCA = preprocessing(hydra_FCA, target_column, std_scaler)
hydra_noFCA = preprocessing(hydra_noFCA, target_column, std_scaler)

print(hydra_FCA[target_column].value_counts())


encoding_mapping = {
    "Normal": 0,
    "Nikto": 1,
    "Nmap": 2,
    "Hydra": 3,
    "SQLi": 4,
    "XSS": 5
}

# voting_model = SIDS_pipeline(df_selected, target_column)
voting_model = None



hydra_FCA = hydra_FCA[selected_columns.tolist() + [target_column]]
hydra_noFCA = hydra_noFCA[selected_columns.tolist() + [target_column]]





# nikto_FCA = nikto_FCA[selected_columns.tolist() + [target_column]]
# nikto_noFCA = nikto_noFCA[selected_columns.tolist() + [target_column]]

# nikto_FCA[target_column] = nikto_FCA[target_column].apply(lambda x: "Normal" if (x == 0 or x == '0') else "Nikto")
# nikto_noFCA[target_column] = nikto_noFCA[target_column].apply(lambda x: "Normal" if (x == 0 or x == '0') else "Nikto")

# nmap_FCA = nmap_FCA[selected_columns.tolist() + [target_column]]
# nmap_noFCA = nmap_noFCA[selected_columns.tolist() + [target_column]]

# nmap_FCA[target_column] = nmap_FCA[target_column].apply(lambda x: "Normal" if (x == 0 or x == '0') else "Nmap")
# nmap_noFCA[target_column] = nmap_noFCA[target_column].apply(lambda x: "Normal" if (x == 0 or x == '0') else "Nmap")

# sqlmap_FCA = sqlmap_FCA[selected_columns.tolist() + [target_column]]
# sqlmap_noFCA = sqlmap_noFCA[selected_columns.tolist() + [target_column]]

# sqlmap_FCA[target_column] = sqlmap_FCA[target_column].apply(lambda x: "Normal" if (x == 0 or x == '0') else "SQLi")
# sqlmap_noFCA[target_column] = sqlmap_noFCA[target_column].apply(lambda x: "Normal" if (x == 0 or x == '0') else "SQLi")

# xss_FCA = xss_FCA[selected_columns.tolist() + [target_column]]
# xss_noFCA = xss_noFCA[selected_columns.tolist() + [target_column]]

# xss_FCA[target_column] = xss_FCA[target_column].apply(lambda x: "Normal" if (x == 0 or x == '0') else "XSS")
# xss_noFCA[target_column] = xss_noFCA[target_column].apply(lambda x: "Normal" if (x == 0 or x == '0') else "XSS")

# voting_model= SIDS_pipeline(df_selected, target_column)

# voting_model, encoding_mapping = SIDS_pipeline(df_selected, target_column)


# hydra_FCA_pred, hydra_FCA_pred_normal, hydra_FCA_pred_non_normal = test_SIDS_model(model=None, df_test=hydra_FCA, target_column=target_column, encoding_mapping=encoding_mapping)
# print(hydra_FCA_pred_normal[target_column].value_counts())
# hydra_FCA_pred_AIDS = AIDS_predict(hydra_FCA_pred_normal, target_column)
# generate_final_reports(encoding_mapping, target_column, hydra_FCA, hydra_FCA_pred_normal, hydra_FCA_pred_non_normal, hydra_FCA_pred, hydra_FCA_pred_AIDS, saving_path="hydra_FCA")

hydra_noFCA_pred, hydra_noFCA_pred_normal, hydra_noFCA_pred_non_normal = test_SIDS_model(model=voting_model, df_test=hydra_noFCA, target_column=target_column, encoding_mapping=encoding_mapping)
print(hydra_noFCA_pred_normal[target_column].value_counts())
hydra_noFCA_pred_AIDS = AIDS_predict(hydra_noFCA, target_column)
generate_final_reports(encoding_mapping, target_column, hydra_noFCA, hydra_noFCA_pred_normal, hydra_noFCA_pred_non_normal, hydra_noFCA_pred, hydra_noFCA_pred_AIDS, saving_path="hydra_noFCA")