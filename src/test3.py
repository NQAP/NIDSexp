import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------- CFMU ----------
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


# ---------- SGU ----------
def build_sgu_for_class(feature_dim=42, cfmu_dim=32, num_minor=20, attn_dim=32):
    """
    SGU for a single class
    Inputs:
        x_cfmu: (batch_size, cfmu_dim)
        x_orig: (batch_size, feature_dim, num_minor) or (feature_dim, num_minor)
    Output:
        generated sample: (batch_size, feature_dim)
    """
    # Inputs
    x_cfmu = layers.Input(shape=(cfmu_dim,), name="x_cfmu")
    x_orig = layers.Input(shape=(feature_dim, num_minor), name="x_orig")

    # 判斷是否有 batch 維度，自動 broadcast
    def expand_x_orig(x):
        x_shape = tf.shape(x)
        if len(x.shape) == 2:  # (feature_dim, num_minor)
            # expand batch dimension
            x = tf.expand_dims(x, axis=0)
            x = tf.repeat(x, tf.shape(x_cfmu)[0], axis=0)
        return x
    x_orig_exp = layers.Lambda(expand_x_orig)(x_orig)

    # --- Attention 部分 ---
    # 將 x_orig 轉置成 (batch_size, num_minor, feature_dim)
    x_orig_T = layers.Permute((2,1))(x_orig_exp)

    # 投影成 key/value
    key   = layers.Dense(attn_dim)(x_orig_T)
    value = layers.Dense(attn_dim)(x_orig_T)

    # query 來自 x_cfmu
    query = layers.Dense(attn_dim)(x_cfmu)

    # 計算注意力分數
    attn_score = layers.Dot(axes=-1)([key, query])
    attn_score = layers.Softmax()(attn_score)

    # 加權求和
    attn_out = layers.Dot(axes=1)([attn_score, value])

    # Dense + Softmax
    attn_out = layers.Dense(feature_dim, activation="softmax")(attn_out)

    # RepeatVector & Permute
    attn_matrix = layers.RepeatVector(num_minor)(attn_out)
    attn_matrix = layers.Permute((2,1))(attn_matrix)

    # 加權原始資料
    weighted_orig = layers.Multiply()([x_orig_exp, attn_matrix])
    x_out = layers.Lambda(lambda t: tf.reduce_sum(t, axis=-1))(weighted_orig)

    return Model([x_cfmu, x_orig], x_out, name="SGU")


# ---------- 完整 Sample Generator Pipeline ----------
def build_sample_generator(noise_dim=32, label_dim=8, feature_dim=42, num_minor=20, attn_dim=32):
    """完整生成器: (noise+label, x_orig) -> generated sample"""
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
    num_minor = [20, 40]
    batch_size = 5
    num_class = 2

    # 建立生成器
    gen_by_class = []
    for i in range(num_class):
        gen = build_sample_generator(noise_dim, label_dim, feature_dim, num_minor[i])
        gen.summary()

    # 多組 noise+label
    noise = np.random.rand(batch_size, noise_dim).astype(np.float32)
    label = np.eye(label_dim)[np.random.choice(label_dim, batch_size)]
    z_in = np.concatenate([noise, label], axis=1)

    # 單一 x_orig
    x_orig_single = np.random.rand(feature_dim, num_minor).astype(np.float32)

    # 擴展成 batch
    x_orig_batch = np.expand_dims(x_orig_single, axis=0)      # (1, feature_dim, num_minor)
    x_orig_batch = np.repeat(x_orig_batch, batch_size, axis=0) # (batch_size, feature_dim, num_minor)

    # Forward pass
    output = gen.predict([z_in, x_orig_batch])

    print("noise+label shape:", z_in.shape)
    print("x_orig shape (single):", x_orig_single.shape)
    print("Output shape:", output.shape)  # (batch_size, feature_dim)
    print("Output sample:\n", output)
