from arlhnids_preprocessing import DataPreprocessor
import numpy as np

# 建立前處理器
prep = DataPreprocessor(label_col="label", scaling="minmax")

# 執行完整 pipeline
X_train, X_test, y_train, y_test = prep.preprocess("./dataset/combined.csv")

print("訓練集大小:", X_train.shape, " 測試集大小:", X_test.shape)
print("類別分布:", dict(zip(*np.unique(y_train, return_counts=True))))
