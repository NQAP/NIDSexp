import numpy as np
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import dense_net as nt
from tqdm import tqdm

# 假設 cfmu_gen 已建立
cfmu_gen = nt.build_cfmu()

# -------------------- 設定參數 --------------------
noise_dim = 32
num_classes = 8
feature_dim = 32
batch_size = 64
epochs = 2000
learning_rate = 0.001

optimizer = Adam(learning_rate)
loss_fn = CategoricalCrossentropy()

# -------------------- 小分類器 --------------------
cfmu_input = Input(shape=(feature_dim,))
pred_label = Dense(num_classes, activation='softmax')(cfmu_input)
classifier = Model(cfmu_input, pred_label)

# -------------------- 預訓練迴圈 --------------------
for epoch in tqdm(range(epochs), desc="CFMU Pretraining"):
    noise_batch = np.random.normal(0, 1, size=(batch_size, noise_dim)).astype(np.float32)
    label_idx = np.random.randint(0, num_classes, size=batch_size)
    label_onehot = np.zeros((batch_size, num_classes), dtype=np.float32)
    label_onehot[np.arange(batch_size), label_idx] = 1

    noise_tf = tf.convert_to_tensor(noise_batch, dtype=tf.float32)
    label_tf = tf.convert_to_tensor(label_onehot, dtype=tf.float32)

    with tf.GradientTape() as tape:
        cfmu_out = cfmu_gen([noise_tf, label_tf], training=True)
        pred = classifier(cfmu_out, training=True)
        loss = loss_fn(label_tf, pred)

    grads = tape.gradient(loss, cfmu_gen.trainable_variables)
    optimizer.apply_gradients(zip(grads, cfmu_gen.trainable_variables))

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.numpy():.6f}")

# -------------------- 儲存模型 --------------------
cfmu_gen.save("./model/cfmu_model/pretrained_cfmu_label_guided.h5")
print("CFMU 預訓練完成並已儲存！")

# -------------------- PCA 可視化 --------------------
num_samples_viz = 100  # 每個 label 測試數量
cfmu_features_all = []

for label_idx in range(num_classes):
    noise_test = np.random.normal(0, 1, size=(num_samples_viz, noise_dim)).astype(np.float32)
    label_onehot = np.zeros((num_samples_viz, num_classes), dtype=np.float32)
    label_onehot[np.arange(num_samples_viz), label_idx] = 1

    noise_tf = tf.convert_to_tensor(noise_test, dtype=tf.float32)
    label_tf = tf.convert_to_tensor(label_onehot, dtype=tf.float32)

    cfmu_out = cfmu_gen([noise_tf, label_tf], training=False).numpy()
    cfmu_features_all.append(cfmu_out)

all_features = np.vstack(cfmu_features_all)
labels_plot = np.repeat(np.arange(num_classes), num_samples_viz)

pca = PCA(n_components=2)
all_features_2d = pca.fit_transform(all_features)

plt.figure(figsize=(8,6))
for i in range(num_classes):
    idx = labels_plot == i
    plt.scatter(all_features_2d[idx, 0], all_features_2d[idx, 1], label=f'Class {i}', alpha=0.6)
plt.title("CFMU Features for Different Labels (PCA 2D)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.show()
