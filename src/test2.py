import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------- 你的 CFMU ----------
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


# ---------- 改良後的 SGU ----------
def build_sgu_for_class(feature_dim=42, cfmu_dim=32, num_minor=20, attn_dim=32):
    """
    SGU for a single class (num_minor samples)
    Inputs:
        x_cfmu: (cfmu_dim,) 
        x_orig: (feature_dim, num_minor)
    Output:
        generated sample: (feature_dim,)
    """
    # Inputs
    x_cfmu = layers.Input(shape=(cfmu_dim,), name="x_cfmu")
    x_orig = layers.Input(shape=(feature_dim, num_minor), name="x_orig")

    # --- Attention 部分 ---
    # 將 x_orig 轉置成 (num_minor, feature_dim)
    x_orig_T = layers.Permute((2,1))(x_orig)  # (num_minor, feature_dim)

    # 投影成 key/value
    key   = layers.Dense(attn_dim)(x_orig_T)   # (num_minor, attn_dim)
    value = layers.Dense(attn_dim)(x_orig_T)   # (num_minor, attn_dim)

    # query 來自 x_cfmu
    query = layers.Dense(attn_dim)(x_cfmu)     # (attn_dim,)

    # 計算注意力分數
    attn_score = layers.Dot(axes=-1)([key, query])  # (num_minor,)
    attn_score = layers.Softmax()(attn_score)       # (num_minor,)

    # 加權求和
    attn_out = layers.Dot(axes=1)([attn_score, value])  # (attn_dim,)

    # --- Dense + Softmax (feature_dim 維度) ---
    attn_out = layers.Dense(feature_dim, activation="softmax")(attn_out)  # (feature_dim,)

    # --- RepeatVector ---
    attn_matrix = layers.RepeatVector(num_minor)(attn_out)  # (num_minor, feature_dim)
    attn_matrix = layers.Permute((2,1))(attn_matrix)        # (feature_dim, num_minor)

    # --- 和 x_orig 結合 ---
    weighted_orig = layers.Multiply()([x_orig, attn_matrix])  # (feature_dim, num_minor)
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=-1))(weighted_orig)  # (feature_dim,)

    return Model([x_cfmu, x_orig], x, name="SGU")


# ---------- 串接 Pipeline ----------
def build_sample_generator(noise_dim=32, label_dim=8, feature_dim=42, num_minor=20, attn_dim=32):
    """完整的生成器: (noise+label, x_orig) -> generated sample"""
    # Inputs
    z_in = layers.Input(shape=(noise_dim+label_dim,), name="noise_label_input")
    x_orig = layers.Input(shape=(feature_dim, num_minor), name="x_orig")

    # CFMU
    cfmu = build_cfmu(noise_dim, label_dim)
    x_cfmu = cfmu(z_in)

    # SGU
    sgu = build_sgu_for_class(feature_dim, cfmu.output_shape[-1], num_minor, attn_dim)
    x_gen = sgu([x_cfmu, x_orig])

    return Model([z_in, x_orig], x_gen, name="SampleGenerator")

if __name__ == "__main__":
    noise_dim = 32
    label_dim = 8
    feature_dim = 42
    num_minor = 20
    batch_size = 4

    # 建立整體模型
    gen = build_sample_generator(noise_dim, label_dim, feature_dim, num_minor)
    gen.summary()

    # 假資料
    noise = np.random.rand(batch_size, noise_dim).astype(np.float32)
    label = np.eye(label_dim)[np.random.choice(label_dim, batch_size)]  # one-hot
    z_in = np.concatenate([noise, label], axis=1)

    x_orig = np.random.rand(batch_size, feature_dim, num_minor).astype(np.float32)

    # Forward pass
    output = gen.predict([z_in, x_orig])

    print("Input (noise+label) shape:", z_in.shape)
    print("Input (x_orig) shape:", x_orig.shape)
    print("Output shape:", output.shape)  # (batch_size, feature_dim)
    print("Output sample:\n", output)