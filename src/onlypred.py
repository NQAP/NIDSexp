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

df_selected = pd.read_csv("./inter_data/selected_feat_0.csv")
df_train, df_test = train_test_split(df_selected, test_size=0.2, random_state=42)


df_SIDS_report, SIDS_pred, df_norm_pred, df_pred_non_norm = SIDS_pipeline(df_train=df_train, df_test=df_test, n_trials=10)

# report, AIDS_pred = anomaly_detection_pipeline_binary(df_selected, episodes=5)

# df_norm_pred = pd.read_csv("./results/prediction/normal.csv")
# df_pred_non_norm = pd.read_csv("./results/prediction/non_normal.csv")


# df_norm_pred.info()
# df_norm_pred.drop(df_norm_pred.columns[0],axis=1, inplace=True)

# data = np.load('./results/prediction/SIDS.npz')
# SIDS_pred = data['SIDS_pred']

# data = np.load('./results/prediction/AIDS.npz')
# AIDS_pred = data['AIDS_pred']

report, AIDS_pred = only_predict(df=df_norm_pred)

print ("SIDS_pred = ", SIDS_pred)
print ("AIDS_pred = ", AIDS_pred)

direcPath=os.path.join("./results", "prediction")

if not os.path.exists(direcPath):
    os.makedirs(direcPath)

# resSave=os.path.join(direcPath, 'SIDS')
# np.savez(resSave, SIDS_pred=SIDS_pred)
# recordSave=os.path.join(direcPath, 'AIDS')
# np.savez(recordSave, AIDS_pred=AIDS_pred)




final_pred_multiclass, final_pred_binary = generate_final_reports(SIDS_pred, df_test, df_norm_pred, df_pred_non_norm, AIDS_pred)
