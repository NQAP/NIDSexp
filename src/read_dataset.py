import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
import json

def preprocessing(file):
    """
    read dataset from csv files
    """
    # 讀取 CSV
    df = pd.read_csv(file)
    df.info()

    """
    drop ID and label columns
    """

    df = df.drop(columns=['ID', 'label'])

    """
    Original numeric data features scaling with standardscalar
    """

    # 標準化 X 的數值欄位
    numeric_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    """
    label encoding
    """

    # 找出所有 object 型別欄位
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # 建立 dict 保存每個欄位的對應關係
    encoding_maps = {}

    # 逐一進行 Label Encoding
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoding_maps[col] = {cls: int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
        print(f"\n欄位 {col} 的對應關係： {encoding_maps[col]}")

    # 將轉換後的資料存成新的 csv 檔案
    df.to_csv("./extra_dataset/encoded.csv", index=False, encoding="utf-8-sig")

    # 儲存對應關係到 JSON 檔
    with open("./extra_dataset/label_encodings.json", "w", encoding="utf-8") as f:
        json.dump(encoding_maps, f, ensure_ascii=False, indent=4)

    print("\n✅ 已將 Label Encoding 對應關係儲存到 label_encodings.json")

    target_column = "attack_cat"
    X = df.drop(columns=[target_column])
    Y = df[target_column]

    """
    Clamping extreme value
    """

    # 迴圈每個欄位
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(float)
            median = X[col].median()
            percentile_95 = X[col].quantile(0.95)
            
            # 使用向量化處理替換值
            mask = (X[col] > 10 * median) | (X[col] > percentile_95)
            X.loc[mask, col] = percentile_95

    

    """
    Find majority and minority class
    """

    counts = Y.value_counts()
    Major_class = []
    Minor_class = []
    I_major = 0
    for val, cnt in counts.items():
        if I_major < cnt:
            I_major = cnt
        I_x = cnt
        if (0.5 * I_major) >= (I_major - I_x):
            Major_class.append(val)
        else:
            Minor_class.append(val)
    print (Major_class)
    print (Minor_class)

    filter_column = target_column

    df_majority = df[df[filter_column].isin(Major_class)]
    df_minority = df[df[filter_column].isin(Minor_class)]

    df_majority.to_csv("./extra_dataset/encoded_majority_class.csv", index=False, encoding="utf-8-sig")
    df_minority.to_csv("./extra_dataset/encoded_minority_class.csv", index=False, encoding="utf-8-sig")

    print(df_majority[target_column].value_counts())
    print(df_minority[target_column].value_counts())

    return df_majority, df_minority
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="loading dataset file (.csv)")
    parser.add_argument("file", help="dataset file")

    args = parser.parse_args()

    df_majority, df_minority = preprocessing(args.file)
    