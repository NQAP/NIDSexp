import pandas as pd
import argparse

def merge_csv(file1, file2, output, how="rows"):
    """
    合併兩個 CSV 檔案，並重新建立 ID 欄位
    :param file1: 第一個 CSV 檔案路徑
    :param file2: 第二個 CSV 檔案路徑
    :param output: 輸出的合併檔案路徑
    :param how: "rows" 表示直向合併 (row-wise)，"cols" 表示橫向合併 (column-wise)
    """
    # 讀取 CSV
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 移除 ID 欄位（不分大小寫）
    for df in [df1, df2]:
        for col in df.columns:
            if col.lower() == "id":
                df.drop(columns=[col], inplace=True)

    # 合併
    if how == "rows":
        merged = pd.concat([df1, df2], axis=0, ignore_index=True)
    elif how == "cols":
        merged = pd.concat([df1, df2], axis=1)
    else:
        raise ValueError("how 參數必須是 'rows' 或 'cols'")

    # 輸出
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"✅ 已輸出合併檔案: {output}，共 {len(merged)} 筆資料")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合併兩個 CSV 檔案，並重新建立 ID 欄位")
    parser.add_argument("file1", help="第一個 CSV 檔案")
    parser.add_argument("file2", help="第二個 CSV 檔案")
    parser.add_argument("-o", "--output", default="merged.csv", help="輸出檔案名稱 (預設: merged.csv)")
    parser.add_argument("--how", choices=["rows", "cols"], default="rows", help="合併方式: rows=直向, cols=橫向")

    args = parser.parse_args()
    merge_csv(args.file1, args.file2, args.output, args.how)
