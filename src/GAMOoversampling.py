import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import json
import tqdm


# === GPU 設定 ===
device = "/GPU:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/CPU:0"
print("Using device:", device)

def conditional_feature_mapping_unit(z=32, c=8):
    r_input = layers.Input(shape=(z,), name="random_vector")
    l_input = layers.Input(shape=(c,), name="label_vector")

    combined = layers.Concatenate()([r_input, l_input])   # (batch, z+c)

    # Dense -> 32
    x = layers.Dense(32)(combined)   # (batch, 32)

    # === Attention over feature dimension ===
    # reshape 成 (batch, seq_len=32, 1)
    x_reshaped = layers.Reshape((32, 1))(x)

    # 計算 attention 權重 (softmax over features)
    attn_weights = layers.Dense(1, activation="softmax")(x_reshaped)  # (batch, 32, 1)

    # 加權求和
    x = layers.Multiply()([x_reshaped, attn_weights])
    x = layers.Flatten()(x)   # 回到 (batch, 32)

    # Dense -> 64 + ReLU + BN
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Dense -> 32 + ReLU + BN
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    return models.Model(inputs=[r_input, l_input], outputs=x, name="ConditionalFeatureMappingUnit")

# === 自定義 SGU ===
def SGU_unit_custom(input_dim=32, orig_dim=42, numMinor=100, name="SGU"):
    cfmu_input = layers.Input(shape=(input_dim,), name=f"{name}_cfmu_input")
    original_input = layers.Input(shape=(orig_dim,), name=f"{name}_original_input")
    
    combined = layers.Concatenate()([cfmu_input, original_input])   # (batch, z+c)
    print(combined.shape)
    # Dense 32 + Softmax
    x = layers.Dense(32, activation="softmax")(combined)
    
    # Attention Layer 32
    x_reshaped = layers.Reshape((32,1))(x)
    attn_weights = layers.Dense(1, activation="softmax")(x_reshaped)
    x = layers.Multiply()([x_reshaped, attn_weights])
    x = layers.Flatten()(x)
    
    # Dense numMinor
    x = layers.Dense(numMinor)(x)
    print(x.shape)
    # RepeatVector Layer
    x = layers.RepeatVector(orig_dim)(x)  # shape -> (orig_dim, numMinor)
    
    # Lambda: reduce mean -> shape = orig_dim
    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1), output_shape=(None, 42))(x)
    
    return models.Model(inputs=[cfmu_input, original_input], outputs=x, name=name)

# === 計算生成樣本數 ===
def compute_generated_samples(chi_m, chi_dict):
    results = []
    for cls, chi_i in chi_dict.items():
        chi_g = max(chi_m - chi_i, 0)
        results.append({"Class": cls[0], "χi": chi_i, "χg": chi_g})
    return pd.DataFrame(results)

def GAMOpreprocessing(df_minority):

    z, c = 32, 8

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

    num_samples = X.shape[0]
    original_dim = X.shape[1]

    labels = np.zeros((num_samples, c))
    for i in range(num_samples):
        labels[i, y[target_column][i]] = 1

    cfmu_model = conditional_feature_mapping_unit(z=z, c=c)

    # SGU units 根據 numMinor 建立
    sgu_units = {}
    for idx, row in df_chi.iterrows():
        cls = row['Class']
        numMinor = row['χg']
        if numMinor > 0:
            sgu_units[cls] = SGU_unit_custom(input_dim=32, orig_dim=original_dim, numMinor=numMinor, name=f"SGU_{cls}")
    
    full_data, full_labels = generate_full_dataset(X, labels, cfmu_model, sgu_units, df_chi, batch_size=8)
    

# === 生成 synthetic dataset 並合併原始 dataset ===
def generate_full_dataset(original_dataset, labels, cfmu_model, sgu_units, df_chi, batch_size=1024):
    """
    原始 dataset: np.array, shape=(num_samples, 42)
    labels: one-hot, shape=(num_samples, c)
    """

    z = 32
    c = 8

    synthetic_data_list = []
    synthetic_labels_list = []

    # 對每個少數類別生成 χg sample
    for cls, sgu in sgu_units.items():
        cls_idx = df_chi[df_chi['Class']==cls].index[0]
        chi_g = df_chi.loc[cls_idx, 'χg']
        if chi_g == 0:
            continue
        
        indices = np.where(labels[:, cls_idx]==1)[0]
        if len(indices)==0:
            continue
        original_cls_data = original_dataset.iloc[indices]
        num_orig = original_cls_data.shape[0]

        # 按 batch 生成
        num_orig = original_cls_data.shape[0]

        generated_so_far = 0

        while generated_so_far < chi_g:
            # 每次生成 batch
            batch_size_actual = min(batch_size, num_orig, chi_g - generated_so_far)
            batch_idx = np.random.choice(num_orig, batch_size_actual, replace=True)
            batch_data = original_cls_data.iloc[batch_idx]

            batch_label = np.zeros((batch_size_actual, c), dtype=np.float32)
            batch_label[:, cls_idx] = 1
            r_vectors = np.random.rand(batch_size_actual, z).astype(np.float32)

            with tf.device(device):
                r_tf = tf.convert_to_tensor(r_vectors, dtype=tf.float32)
                label_tf = tf.convert_to_tensor(batch_label, dtype=tf.float32)
                orig_tf = tf.convert_to_tensor(batch_data, dtype=tf.float32)
                cfmu_out = cfmu_model([r_tf, label_tf])
                generated = sgu([cfmu_out, orig_tf]).numpy()

            synthetic_data_list.append(generated)
            print(generated)
            synthetic_labels_list.extend([cls]*generated.shape[0])
            generated_so_far += generated.shape[0]

    # 合併所有 synthetic data
    if synthetic_data_list:
        synthetic_data = np.vstack(synthetic_data_list)
        synthetic_labels = np.array(synthetic_labels_list)
    else:
        synthetic_data = np.array([])
        synthetic_labels = np.array([])

    # 合併原始 dataset
    full_data = np.vstack([original_dataset, synthetic_data])
    full_labels = np.vstack([labels, 
                             np.eye(c)[[df_chi[df_chi['Class']==lbl].index[0] for lbl in synthetic_labels]]])
    
    return full_data, full_labels



