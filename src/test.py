import tensorflow as tf
from tensorflow.keras import layers, Model, Input
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


# ---------------- SGU ---------------
def build_sgu_for_class(feature_dim=42, cfmu_dim=32, num_minor=20, attn_dim=32):
    """
    SGU for a single class (num_minor samples)
    Inputs:
        x_cfmu: (cfmu_dim,) 
        x_orig: (feature_dim, num_minor)
    Output:
        generated sample: (feature_dim,)
    """

    # Input
    x_cfmu = Input(shape=(cfmu_dim,), name="x_cfmu")             # (cfmu_dim,)
    x_orig = Input(shape=(feature_dim, num_minor), name="x_orig") # (feature_dim, num_minor)

    # --- Attention ---
    query = layers.Dense(attn_dim)(x_cfmu)      # (attn_dim,)
    key   = layers.Dense(attn_dim)(x_orig)      # (feature_dim, attn_dim)
    value = layers.Dense(attn_dim)(x_orig)      # (feature_dim, attn_dim)

    # Attention score
    attn_score = layers.Lambda(lambda t: tf.matmul(t[0:1], t[1], transpose_b=True))([query, key])
    attn_score = layers.Softmax(axis=-1)(attn_score)
    attn_out = layers.Lambda(lambda t: tf.matmul(t[0], t[1]))([attn_score, value])
    attn_out = layers.Reshape((attn_dim,))(attn_out)

    # --- Dense(softmax, 42) ---
    d = layers.Dense(feature_dim, activation="softmax")(attn_out)  # (42,)

    # RepeatVector -> (feature_dim, num_minor)
    r = layers.RepeatVector(num_minor)(d)                # (num_minor, 42)
    r = layers.Lambda(lambda x: tf.transpose(x, perm=[1,0]), output_shape=(feature_dim, num_minor))(r)  # (42, num_minor)

    # Multiply with x_orig
    mul = layers.Multiply()([r, x_orig])                # (42, num_minor)

    # Mean over num_minor axis -> (feature_dim,)
    out = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1), output_shape=(feature_dim, ))(mul)

    return Model([x_cfmu, x_orig], out, name="SGU_class")

# ---------------- 測試 ----------------
if __name__ == "__main__":
    feature_dim = 42
    cfmu_dim = 32
    noise_dim = 32
    label_dim = 8
    num_minor = 100

    # 建立 CFMU 與 SGU
    cfmu = build_cfmu(noise_dim=noise_dim, label_dim=label_dim)
    sgu  = build_sgu_for_class(feature_dim=feature_dim, cfmu_dim=cfmu_dim)

    sgu.summary()

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

    print(X_cfmu.shape)

    # 用 SGU 生成資料
    X_sgu = sgu.predict([X_orig, X_cfmu])
    print("SGU output shape:", X_sgu[0], X_sgu[1])  # 預期 (100, 42)
