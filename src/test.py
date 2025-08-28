import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


# ---------------- CFMU ----------------
def build_cfmu(noise_dim=32, label_dim=8):
    """Conditional Feature Mapping Unit"""
    x_in = layers.Input(shape=(noise_dim+label_dim,), name="cfmu_input")

    # Dense 32
    x = layers.Dense(32)(x_in)

    # Attention Layer
    query = layers.Dense(32)(x)
    key   = layers.Dense(32)(x)
    value = layers.Dense(32)(x)

    attn_score = layers.Dot(axes=-1)([query, key])
    attn_score = layers.Softmax()(attn_score)
    attn_out   = layers.Multiply()([attn_score, value])

    # Dense 64 + ReLU
    x = layers.Dense(64, activation="relu")(attn_out)
    # Dense 32 + ReLU
    x = layers.Dense(32, activation="relu")(x)

    return Model(x_in, x, name="CFMU")


# ---------------- SGU ----------------
def build_sgu_dataset(feature_dim=42, cfmu_dim=32):
    """
    SGU: Sample Generation Unit (dataset version)
    Input:
        - 原始資料: (num_minor, feature_dim)
        - CFMU輸出: (num_minor, cfmu_dim)
    Output:
        - 生成資料: (num_minor, feature_dim)
    """

    # 兩個輸入 (批次大小為 num_minor)
    x_orig = layers.Input(shape=(feature_dim,), name="sgu_original_input")
    x_cfmu = layers.Input(shape=(cfmu_dim,), name="sgu_cfmu_input")

    # Attention Layer
    query = layers.Dense(32)(x_orig)
    key   = layers.Dense(32)(x_cfmu)
    value = layers.Dense(32)(x_cfmu)

    attn_score = layers.Dot(axes=-1)([query, key])
    attn_score = layers.Softmax()(attn_score)
    attn_out   = layers.Multiply()([attn_score, value])

    # Dense + Softmax
    h = layers.Dense(32, activation="softmax")(attn_out)

    # Dense → RepeatVector → Lambda (生成新的 minority data)
    d = layers.Dense(feature_dim, activation="relu")(h)
    r = layers.RepeatVector(1)(d)   # (num_minor, 1, feature_dim)
    out = layers.Lambda(lambda z: tf.squeeze(z, axis=1))(r)  # (num_minor, feature_dim)

    return Model([x_orig, x_cfmu], out, name="SGU_dataset")

# ---------------- 測試 ----------------
if __name__ == "__main__":
    feature_dim = 42
    cfmu_dim = 32
    noise_dim = 32
    label_dim = 8
    num_minor = 100

    # 建立 CFMU 與 SGU
    cfmu = build_cfmu(noise_dim=noise_dim, label_dim=label_dim)
    sgu  = build_sgu_dataset(feature_dim=feature_dim, cfmu_dim=cfmu_dim)

    # 原始 minority data
    X_orig = np.random.rand(num_minor, feature_dim).astype(np.float32)

    # CFMU 輸入 (noise + one-hot label)
    noise = np.random.rand(num_minor, noise_dim).astype(np.float32)

    # 隨機生成 label indices，轉成 one-hot
    label_indices = np.random.randint(0, label_dim, size=(num_minor,))
    labels_onehot = np.eye(label_dim)[label_indices].astype(np.float32)

    X_cfmu_input = np.concatenate([noise, labels_onehot], axis=1)

    # 用 CFMU 生成特徵
    X_cfmu = cfmu.predict(X_cfmu_input)

    # 用 SGU 生成資料
    X_sgu = sgu.predict([X_orig, X_cfmu])
    print("SGU output shape:", X_sgu[0], X_sgu[1])  # 預期 (100, 42)
