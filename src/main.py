from preprocessing import preprocessing
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="loading dataset file (.csv)")
    parser.add_argument("file", help="dataset file")

    args = parser.parse_args()

    df_majority, df_minority = preprocessing(args.file)
    # GAMOpreprocessing(df_minority)

    target_column = "attack_cat"
    X = df_minority.drop(columns=[target_column])
    print("X.shape:", X.shape)
    Y = df_minority[target_column]

    df_minority_train, df_minority_test = train_test_split(df_minority, test_size=0.2, random_state=42)

    df_minority_train.to_csv("./extra_dataset/df_minority_train.csv",index=False)
    df_minority_test.to_csv("./extra_dataset/df_minority_test.csv",index=False)
    
    
