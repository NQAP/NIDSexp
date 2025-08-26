from preprocessing import preprocessing
# from GAMOoversampling import GAMOpreprocessing
from GAMOdemo import Generator, GAMOpreprocessing
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="loading dataset file (.csv)")
    parser.add_argument("file", help="dataset file")

    args = parser.parse_args()

    df_majority, df_minority = preprocessing(args.file)
    # GAMOpreprocessing(df_minority)

    target_column = "attack_cat"
    X = df_minority.drop(columns=[target_column])
    Y = df_minority[target_column]
    
    num_classes = 8
    output_dim = X.shape[1]
    samples_per_class = [0, 15209, 21129, 22902, 31393, 31647, 32260, 33263]

    # 模擬原始 dataset (每類 1000 筆, dim=42)
    dataset_by_class = [np.random.randn(1000, output_dim) for _ in range(num_classes)]

    gen = Generator(z_dim=32, num_classes=num_classes,
                    output_dim=output_dim,
                    samples_per_class=samples_per_class,
                    device="/GPU:0")

    fake_all = gen.generate_all(dataset_by_class, batch_size=256)
    print("生成完成:", fake_all.shape)
    print(fake_all[0:3])
