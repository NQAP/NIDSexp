import pandas as pd
import json
import argparse

def decode_labels(encoded_file="./dataset/encoded.csv", mapping_file="./dataset/label_encodings.json", output_file="./dataset/decoded.csv"):
    """
    將 DataFrame 中的數值欄位還原成文字標籤
    :param df: 已經 Label Encoding 過的 DataFrame
    :param mapping_file: 存有 encoding map 的 JSON 檔
    :return: 還原後的 DataFrame
    """
    # 讀取 JSON
    with open(mapping_file, "r", encoding="utf-8") as f:
        encoding_maps = json.load(f)

    # 讀取 csv
    df = pd.read_csv(encoded_file)

    # 逐一還原欄位
    for col, mapping in encoding_maps.items():
        reverse_map = {v: k for k, v in mapping.items()}
        df[col] = df[col].map(reverse_map)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print("\n還原成功")


# 範例：讀取剛剛編碼過的 CSV
if __name__ == "__main__":
    # 假設你前面已經存了 encoded.csv
    parser = argparse.ArgumentParser(description="decode encoded file (csv) with your encoding code (json)")
    parser.add_argument("encoded", help="encoded csv file")
    parser.add_argument("label", help="label encoding json file")
    parser.add_argument("-o", "--output",default="./dataset/decoded.csv", help="output file path")

    args = parser.parse_args()

    # 還原
    decode_labels(args.encoded, args.label, args.output)
