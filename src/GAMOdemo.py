import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import json

# =====================
# Conditional Feature Mapping Unit (CFMU)
# =====================
def build_cfm_unit(z_dim=32, num_classes=8):
    input_noise = Input(shape=(z_dim,))
    input_label = Input(shape=(num_classes,))

    x = layers.Concatenate()([input_noise, input_label])  # (40,)
    x = layers.Dense(32)(x)

    # Attention block
    q = layers.Dense(32)(x)
    k = layers.Dense(32)(x)
    v = layers.Dense(32)(x)

    q = layers.Reshape((1, 32))(q)
    k = layers.Reshape((1, 32))(k)
    v = layers.Reshape((1, 32))(v)

    attn_out = layers.Attention()([q, v, k])  # (batch, 1, 32)
    x = layers.Flatten()(attn_out)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    return models.Model([input_noise, input_label], x, name="CFMU")


# =====================
# Sample Generation Unit (SGU)
# =====================
def build_sgu(output_dim=42, num_samples=1000):
    fake_input = Input(shape=(32,))
    real_input = Input(shape=(output_dim,))

    x = layers.Concatenate()([fake_input, real_input])

    x = layers.Dense(32, activation="softmax")(x)

    # Attention block
    q = layers.Dense(32)(x)
    k = layers.Dense(32)(x)
    v = layers.Dense(32)(x)

    q = layers.Reshape((1, 32))(q)
    k = layers.Reshape((1, 32))(k)
    v = layers.Reshape((1, 32))(v)

    attn_out = layers.Attention()([q, v, k])
    x = layers.Flatten()(attn_out)

    # Dense -> Repeat -> Lambda
    x = layers.Dense(num_samples)(x)
    x = layers.RepeatVector(output_dim)(x)  # (output_dim, num_samples)
    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1))(x)  # (output_dim,)

    return models.Model([fake_input, real_input], x, name=f"SGU_{num_samples}")


# =====================
# Generator (CFMU + SGU)
# =====================
class Generator:
    def __init__(self, z_dim=32, num_classes=8, output_dim=42, samples_per_class=None, device="/GPU:0"):
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.samples_per_class = samples_per_class or [100] * num_classes

        with tf.device(device):
            self.cfm = build_cfm_unit(z_dim, num_classes)
            self.sgu_list = [
                build_sgu(output_dim=output_dim, num_samples=n)
                for n in self.samples_per_class
            ]

    def generate_for_class(self, class_id, real_data, batch_size=128):
        """逐 batch 生成指定類別的資料"""
        num_samples = self.samples_per_class[class_id]
        generated = []

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            bsz = end - start

            # noise + one-hot label
            z = np.random.normal(size=(bsz, self.z_dim))
            labels = np.zeros((bsz, self.num_classes))
            labels[:, class_id] = 1

            # CFMU
            fake_feat = self.cfm.predict([z, labels], verbose=0)

            # 隨機取 real data batch
            idx = np.random.choice(len(real_data), size=bsz)
            real_batch = real_data[idx]

            # SGU
            fake_batch = self.sgu_list[class_id].predict([fake_feat, real_batch], verbose=0)

            generated.append(fake_batch)

        return np.vstack(generated)

    def generate_all(self, dataset_by_class, batch_size=128):
        """針對所有類別生成資料"""
        all_generated = []
        for i in range(self.num_classes):
            print(f"Generating for class {i}, target {self.samples_per_class[i]} samples...")
            if self.samples_per_class[i] <= 0:
                print(f"Class {i} doesn't need to generate...")
                continue
            fake_data = self.generate_for_class(i, dataset_by_class[i], batch_size)
            all_generated.append(fake_data)

        return np.vstack(all_generated)


# === 計算生成樣本數 ===
def compute_generated_samples(chi_m, chi_dict):
    results = []
    for cls, chi_i in chi_dict.items():
        chi_g = max(chi_m - chi_i, 0)
        results.append({"Class": cls[0], "χi": chi_i, "χg": chi_g})
    return pd.DataFrame(results)

def GAMOpreprocessing(df_minority):

    target_column = "attack_cat"
    X = df_minority.drop(columns=[target_column])
    Y = df_minority[target_column]

    chi_m = 33393
    chi_dict = {
        "Exploits": 33393,
        "Fuzzers": 18184,
        "DoS": 12264,
        "Reconnaissance": 10491,
        "Analysis": 2000,
        "Backdoor": 1746,
        "Shellcode": 1133,
        "Worms": 130
    }

    df_chi = compute_generated_samples(chi_m, chi_dict)

    # 建立 dict 保存每個欄位的對應關係
    encoding_maps = {}

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    encoding_maps[target_column] = {cls: int(code) for cls, code in zip(le.classes_, le.transform(le.classes_))}
    print(f"\n欄位 {target_column} 的對應關係： {encoding_maps[target_column]}")


    with open("./extra_dataset/minor_label_encodings.json", "w", encoding="utf-8") as f:
        json.dump(encoding_maps, f, ensure_ascii=False, indent=4)

    y = pd.DataFrame()
    y[target_column] = Y

    counts = y.value_counts()
    chi_m = 33393

    chi_dict = {
        "Exploits": 33393,
        "Fuzzers": 18184,
        "DoS": 12264,
        "Reconnaissance": 10491,
        "Analysis": 2000,
        "Backdoor": 1746,
        "Shellcode": 1133,
        "Worms": 130
    }
    
    for val, cnt in counts.items():
        chi_dict[val] = max(chi_m - cnt, 0)

    num_classes = y.shape[0]
    output_dim = X.shape[1]
    samples_per_class = [0, 15209, 21129, 22902, 31393, 31647, 32260, 33263]

    # 模擬原始 dataset (每類 1000 筆, dim=42)
    dataset_by_class = [np.random.randn(1000, output_dim) for _ in range(num_classes)]

    gen = Generator(z_dim=32, num_classes=num_classes,
                    output_dim=output_dim,
                    samples_per_class=samples_per_class,
                    device="/GPU:0")

    fake_all = gen.generate_all(dataset_by_class, batch_size=256)

# =====================
# 測試用範例
# =====================
if __name__ == "__main__":
    num_classes = 8
    output_dim = 42
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