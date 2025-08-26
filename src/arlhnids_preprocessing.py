# arlhnids_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, label_col="label", scaling="minmax"):
        """
        :param label_col: 標籤欄位名稱
        :param scaling: 選擇 "minmax" 或 "standard"
        """
        self.label_col = label_col
        self.scaling = scaling
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler() if scaling == "minmax" else None

    def load_dataset(self, path):
        """讀取 CSV 資料"""
        df = pd.read_csv(path)
        return df

    def clean_features(self, df):
        """移除缺失值與常數特徵"""
        df = df.dropna()
        nunique = df.nunique()
        const_cols = nunique[nunique <= 1].index
        df = df.drop(columns=const_cols)
        return df

    def encode_labels(self, df):
        """標籤編碼"""
        y = self.label_encoder.fit_transform(df[self.label_col].astype(str))
        X = df.drop(columns=[self.label_col]).astype(float)
        return X, y

    def scale_features(self, X):
        """特徵縮放"""
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled
        return X.values

    def split_dataset(self, X, y, test_size=0.2, random_state=42):
        """切分 Train/Test，分層抽樣"""
        return train_test_split(X, y, test_size=test_size, 
                                random_state=random_state, stratify=y)

    def preprocess(self, path, test_size=0.2, random_state=42):
        """完整流程"""
        df = self.load_dataset(path)
        df = self.clean_features(df)
        X, y = self.encode_labels(df)
        X_scaled = self.scale_features(X)
        return self.split_dataset(X_scaled, y, test_size, random_state)
