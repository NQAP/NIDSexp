# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import os
import numpy as np
import dense_suppli as spp
import dense_net as nt
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle

# For selecting a GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_optimizer():
    return Adam(5e-4, 0.5)

# 確保啟用 eager execution
tf.config.run_functions_eagerly(True)

# Ground works
fileName=['./extra_dataset/df_minority_train.csv', './extra_dataset/df_minority_test.csv']
fileStart='UBSW_NB15_Gamo_New2'
fileEnd, savePath='_Model.h5', fileStart+'/'
latDim, modelSamplePd, resSamplePd=32, 500, 100

batchSize, max_step=512, 5000

trainS, labelTr=spp.fileRead(fileName[0])
testS, labelTs=spp.fileRead(fileName[1])

n, m=trainS.shape[0], testS.shape[0]

labelTr, labelTs, c, pInClass, classMap, label_dict=spp.relabel(labelTr, labelTs)

# 儲存對應關係到 JSON 檔
with open("./extra_dataset/minor_label_encodings.json", "w", encoding="utf-8") as f:
    json.dump(label_dict, f, ensure_ascii=False, indent=4)

imbalancedCls, toBalance, imbClsNum, ir=spp.irFind(pInClass, c)

# ------------------------------
# Optimizer
# ------------------------------
optimizer_G = []
for i in range(imbClsNum):
    optimizer_G.append(Adam(2e-4, 0.5))
optimizer_D = Adam(1e-4, 0.5)
optimizer_M = Adam(1e-3, 0.5)

labelsCat=to_categorical(labelTr)

shuffleIndex=np.random.choice(np.arange(n), size=(n,), replace=False)
trainS=trainS[shuffleIndex]
labelTr=labelTr[shuffleIndex]
labelsCat=labelsCat[shuffleIndex]
classMap=list()
for i in range(c):
    classMap.append(np.where(labelTr==i)[0])

# model initialization
mlp=nt.denseMlpCreate()
mlp.compile(loss='categorical_crossentropy', optimizer=optimizer_M)

dis=nt.denseDisCreate()
dis.compile(loss='mean_squared_error', optimizer=optimizer_D)

cfmu_gen = load_model(
    "./model/cfmu_model/pretrained_cfmu_label_guided.h5", 
    custom_objects={"SelfAttention": nt.SelfAttention}
)
cfmu_gen.trainable = False  # 鎖定權重

gen = []
for i in range(imbClsNum):
    dataMinor=trainS[classMap[i], :]
    numMinor=dataMinor.shape[0]
    gen.append(nt.denseGamoGenCreate(latDim, numMinor, dataMinor))
    gen[i].compile(optimizer=optimizer_G[i])

print(pInClass)

def adjust_pi(O3_dict, O6_dict, P_i_dict, target=0, lr=0.1, min_val=1e-3, max_val=1.0):
    """
    自動微調 P_i，使 (O3 - O6) 接近 target
    
    Args:
        O3_dict: dict, 每個類別目前的 O3 值
        O6_dict: dict, 每個類別目前的 O6 值
        P_i_dict: dict, 當前 P_i 值
        target: float, 希望 O3-O6 接近的目標值
        lr: float, 調整步長
        min_val, max_val: P_i 的範圍限制
    Returns:
        new_P_i_dict: dict, 更新後的 P_i 值
    """
    new_P_i_dict = P_i_dict
    for i in O3_dict.keys():
        diff = (O3_dict[i] - O6_dict[i]) - target
        # 若 O3-O6 太大，減小 P_i；太小，增加 P_i
        new_val = P_i_dict[i] * (1 - lr * np.sign(diff))
        # 限制在 min_val ~ max_val
        new_val = np.clip(new_val, min_val, max_val)
        new_P_i_dict[i] = new_val
    # 重新正規化 P_i，使總和為 1
    total = sum(new_P_i_dict.values())
    for i in new_P_i_dict.keys():
        new_P_i_dict[i] /= total
    return new_P_i_dict


num_classes = len(np.unique(labelTr))

# 你的原本 P_i_dict / P_c_dict 保留
P_i_dict = {i: (1.0/N_i) / sum(1.0/np.array(pInClass)) for i, N_i in enumerate(pInClass)}
P_i_sum = sum(P_i_dict.values())
P_i_dict = {i: P_i_dict[i] / P_i_sum for i in range(c)}  
P_c_dict = {i: 1.0 for i in range(c)}

# 額外新增：MLP 的 class_weights (平滑過的 inverse frequency)
alpha = 0.3  # 平滑指數，避免極端
counts = np.array(pInClass, dtype=np.float32)
raw = 1.0 / (np.power(counts, alpha) + 1e-12)
class_weights = raw / raw.sum() * len(counts)
max_weight = 20.0
class_weights = np.minimum(class_weights, max_weight)
class_weights_tf = tf.constant(class_weights, dtype=tf.float32)

print("class_weights (for MLP):", class_weights)

# for record saving
# ---------------- Stratified Batch Division ----------------
batchDiv, numBatches, bSStore, batch_data_idx = spp.stratified_batchDivision(
    labels=labelTr, 
    batchSize=batchSize, 
    numClasses=c, 
    min_samples_per_class=15
)

genClassPoints = int(np.ceil(batchSize / c))

if not os.path.exists(fileStart):
    os.makedirs(fileStart)

iter=int(np.ceil(max_step/resSamplePd)+1)
acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
confMatSaveTr, confMatSaveTs=np.zeros((iter, c, c)), np.zeros((iter, c, c))
tprSaveTr, tprSaveTs=np.zeros((iter, c)), np.zeros((iter, c))

# # Class-wise warm-up MLP
# ce_none = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

# warmup_epochs_M = 20
# for e in range(warmup_epochs_M):
#     epoch_class_loss = np.zeros(c, dtype=np.float32)

#     for j in range(numBatches):
#         chosen = batch_data_idx[j]  # stratified batch index
#         x = trainS[chosen]
#         y = labelsCat[chosen]
#         x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
#         y_tf = tf.convert_to_tensor(y, dtype=tf.float32)

#         with tf.GradientTape() as tape:
#             preds = mlp(x_tf, training=True)
#             per_sample = ce_none(y_tf, preds)
#             # sample weights based on class weights
#             sample_weights = tf.reduce_sum(y_tf * class_weights_tf, axis=1)
#             loss = tf.reduce_mean(per_sample * sample_weights)

#         grads = tape.gradient(loss, mlp.trainable_variables)
#         optimizer_M.apply_gradients(zip(grads, mlp.trainable_variables))

#         # 計算每個類別 loss
#         for k in range(c):
#             mask = tf.cast(y_tf[:, k] > 0, tf.float32)
#             if tf.reduce_sum(mask) > 0:
#                 class_loss = tf.reduce_mean(per_sample * mask)
#                 epoch_class_loss[k] += class_loss.numpy()

#     # 平均每個 batch 的 class loss
#     epoch_class_loss /= numBatches

#     print(f"Warm-up Epoch {e+1}/{warmup_epochs_M}")
#     for k in range(c):
#         print(f"  Class {k} CE_loss: {epoch_class_loss[k]:.4f}")


# # ------------------------------
# # Warm-up Discriminator (D)
# # ------------------------------
# warmup_epochs_D = 10
# batch_size_warmup = 64

# for i in range(imbClsNum):  # 每個少數類別分開 warm-up
#     dataMinor = trainS[classMap[i]]
#     num_samples = min(2000, dataMinor.shape[0])  # 最多 2000 筆
#     dataMinor = dataMinor[:num_samples]

#     print(f"Warm-up D for class {i}, num_samples: {num_samples}")

#     for epoch in range(warmup_epochs_D):
#         idx_shuffle = np.random.permutation(num_samples)
#         dataMinor_shuffled = dataMinor[idx_shuffle]

#         for start in range(0, num_samples, batch_size_warmup):
#             end = min(start + batch_size_warmup, num_samples)
#             batch_real = dataMinor_shuffled[start:end]

#             # 對應類別 one-hot
#             batch_labels = np.zeros((batch_real.shape[0], c))
#             batch_labels[:, i] = 1

#             batch_real_tf = tf.convert_to_tensor(batch_real, dtype=tf.float32)
#             batch_labels_tf = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

#             # 這裡只用真實樣本，D 判別為真 (label=1)
#             labels_real = tf.ones((batch_real_tf.shape[0], 1), dtype=tf.float32)

#             with tf.GradientTape() as tape_D:
#                 D_pred = dis([batch_real_tf, batch_labels_tf], training=True)
#                 D_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels_real, D_pred))

#             # Update Discriminator
#             grads = tape_D.gradient(D_loss, dis.trainable_variables)
#             optimizer_D.apply_gradients(zip(grads, dis.trainable_variables))

#         print(f"Class {i} epoch {epoch+1}/{warmup_epochs_D}, D_loss: {D_loss.numpy():.5f}")

# ------------------------------
# Step 1: Pretrain M (Classifier)
# ------------------------------

# 擾動函數
def augment_minority(batch, noise_std=0.01, scale_range=(0.95, 1.05)):
    batch = batch + np.random.normal(0, noise_std, size=batch.shape)
    scale = np.random.uniform(scale_range[0], scale_range[1], size=batch.shape)
    return batch * scale

ce_none = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

epochs_pretrain_M = 50
target_samples = 10000
num_samples_class = np.array([137, 1220, 1847, 2151, 11122, 13098, 19459, 35607])
oversample_times = np.ceil(target_samples / num_samples_class).astype(int)
max_batch_size = 512

# 儲存每個 epoch 的統計
metrics_history = {i: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for i in range(c)}
loss_history = []

for e in range(epochs_pretrain_M):
    batch_loss_list = []
    
    # =========================
    # 1️⃣ 每個 class 單獨訓練
    # =========================
    for i in range(c):
        data_class = trainS[classMap[i]]
        labels_class = labelsCat[classMap[i]]
        num_samples = data_class.shape[0]
        repeat = oversample_times[i]  # 少數類可多抽幾次

        for r in range(repeat):
            idx_shuffle = np.random.permutation(num_samples)
            data_class_shuffled = data_class[idx_shuffle]
            labels_class_shuffled = labels_class[idx_shuffle]

            for start in range(0, num_samples, max_batch_size):
                end = min(start + max_batch_size, num_samples)
                x_batch = data_class_shuffled[start:end]
                y_batch = labels_class_shuffled[start:end]

                # 少數類可以做隨機擾動
                if r > 0:
                    x_batch = augment_minority(x_batch, noise_std=0.01, scale_range=(0.95, 1.05))

                x_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)
                y_tf = tf.convert_to_tensor(y_batch, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    preds = mlp(x_tf, training=True)
                    per_sample = ce_none(y_tf, preds)
                    sample_weights = tf.reduce_sum(y_tf * class_weights_tf, axis=1)
                    loss = tf.reduce_mean(per_sample * sample_weights)

                grads = tape.gradient(loss, mlp.trainable_variables)
                optimizer_M.apply_gradients(zip(grads, mlp.trainable_variables))
                batch_loss_list.append(loss.numpy())

                # ===== 計算 batch class-wise metrics =====
                y_true_batch = np.argmax(y_batch, axis=1)
                y_pred_batch = np.argmax(preds.numpy(), axis=1)

                for j in range(c):
                    mask = (y_true_batch == j)
                    if np.sum(mask) > 0:
                        acc_j = accuracy_score(y_true_batch[mask], y_pred_batch[mask])
                        prec_j = precision_score(y_true_batch[mask], y_pred_batch[mask], average='macro', zero_division=0)
                        rec_j = recall_score(y_true_batch[mask], y_pred_batch[mask], average='macro', zero_division=0)
                        f1_j = f1_score(y_true_batch[mask], y_pred_batch[mask], average='macro', zero_division=0)
                    else:
                        acc_j, prec_j, rec_j, f1_j = 0.0, 0.0, 0.0, 0.0

                    metrics_history[j]['acc'].append(acc_j)
                    metrics_history[j]['prec'].append(prec_j)
                    metrics_history[j]['rec'].append(rec_j)
                    metrics_history[j]['f1'].append(f1_j)

    # =========================
    # 2️⃣ epoch 結束後 mix-batch 訓練
    # =========================
    x_mix_list, y_mix_list = [], []
    mix_batch_size_per_class = 100  # 每個 class 抽樣數，可調整
    for i in range(c):
        data_class = trainS[classMap[i]]
        labels_class = labelsCat[classMap[i]]
        num_samples = min(mix_batch_size_per_class, data_class.shape[0])
        idx = np.random.choice(data_class.shape[0], num_samples, replace=False)
        x_mix_list.append(data_class[idx])
        y_mix_list.append(labels_class[idx])

    x_mix = tf.convert_to_tensor(np.concatenate(x_mix_list, axis=0), dtype=tf.float32)
    y_mix = tf.convert_to_tensor(np.concatenate(y_mix_list, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        preds_mix = mlp(x_mix, training=True)
        per_sample = ce_none(y_mix, preds_mix)
        sample_weights = tf.reduce_sum(y_mix * class_weights_tf, axis=1)
        loss_mix = tf.reduce_mean(per_sample * sample_weights)

    grads = tape.gradient(loss_mix, mlp.trainable_variables)
    optimizer_M.apply_gradients(zip(grads, mlp.trainable_variables))
    batch_loss_list.append(loss_mix.numpy())

    # =========================
    # Epoch 結束，統計每個 class 平均 metrics
    # =========================
    print(f"[M] Epoch {e+1}/{epochs_pretrain_M}, Loss: {np.mean(batch_loss_list):.5f}")
    for j in range(c):
        acc_j = np.mean(metrics_history[j]['acc'][-len(batch_loss_list):])
        prec_j = np.mean(metrics_history[j]['prec'][-len(batch_loss_list):])
        rec_j = np.mean(metrics_history[j]['rec'][-len(batch_loss_list):])
        f1_j = np.mean(metrics_history[j]['f1'][-len(batch_loss_list):])

        print(f"  Class {j} : Acc={acc_j:.4f}, Prec={prec_j:.4f}, "
              f"Rec={rec_j:.4f}, F1={f1_j:.4f}")





# freeze M
mlp.trainable = False

direcPath=savePath+'gamo_models_'
if not os.path.exists(direcPath):
    os.makedirs(direcPath)
mlp.save(direcPath+'/MLP_'+fileEnd)

# ------------------------------
# Step 2: Pretrain D (Discriminator)
# ------------------------------
epochs_pretrain_D = 10

# 先取得少數類 id
minor_class_ids = [i for i in range(c)]

for e in range(epochs_pretrain_D):
    # 隨機打亂 batch 的順序
    idx_shuffle_batches = np.random.permutation(numBatches)

    all_y_true = []
    all_y_pred = []

    for j in idx_shuffle_batches:
        idx_batch = batch_data_idx[j]
        batch_real = trainS[idx_batch]
        batch_labels = labelsCat[idx_batch]

        batch_real_tf = tf.convert_to_tensor(batch_real, dtype=tf.float32)
        batch_labels_tf = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

        D_loss_classes = []

        for i in range(imbClsNum):
            # 篩選 batch 中屬於 class i 的樣本
            class_idx = np.where(np.argmax(batch_labels, axis=1) == i)[0]
            if len(class_idx) == 0:
                D_loss_classes.append(np.float32(0.0))
                continue

            dataMinor_batch = tf.convert_to_tensor(batch_real[class_idx], dtype=tf.float32)
            labels_onehot = tf.convert_to_tensor(batch_labels[class_idx], dtype=tf.float32)

            # 生成對應的 fake sample
            num_samples = tf.shape(dataMinor_batch)[0]
            z = tf.random.normal((num_samples, latDim))
            cfmu_features = cfmu_gen([z, labels_onehot])
            x_fake = gen[i](cfmu_features)

            with tf.GradientTape() as tape_D_i:
                D_real = dis([dataMinor_batch, labels_onehot], training=True)
                D_fake = dis([x_fake, labels_onehot], training=True)

                # 加入 class weight 放大少數類
                w = class_weights_tf[i]

                O5 = w * tf.math.log(D_real + 1e-8)
                O6 = w * tf.math.log(1 - D_fake + 1e-8)
                D_loss_i = -tf.reduce_mean(O5 + O6)

            grads_i = tape_D_i.gradient(D_loss_i, dis.trainable_variables)
            optimizer_D.apply_gradients(zip(grads_i, dis.trainable_variables))
            D_loss_classes.append(D_loss_i.numpy())
            # print (D_loss_i)

            # 收集 batch 預測結果
            all_y_true.extend(np.argmax(batch_labels[class_idx], axis=1))
            all_y_pred.extend(np.argmax(D_real.numpy(), axis=1))

    # Epoch 結束後統計
    acc = accuracy_score(all_y_true, all_y_pred)
    prec = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    rec = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)

    # 少數類檢查
    minor_idx = [idx for idx, cls in enumerate(all_y_true) if cls in minor_class_ids]
    minor_acc = accuracy_score(np.array(all_y_true)[minor_idx], np.array(all_y_pred)[minor_idx])
    minor_f1 = f1_score(np.array(all_y_true)[minor_idx], np.array(all_y_pred)[minor_idx],
                        average='macro', zero_division=0)

    print(f"[D] Epoch {e+1}/{epochs_pretrain_D}")
    print(f"    Loss per class: {D_loss_classes}")
    print(f"    Overall Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"    Minor class Acc: {minor_acc:.4f}, Minor class F1: {minor_f1:.4f}")


# freeze D
dis.trainable = False

dis.save(direcPath+'/DIS_'+fileEnd)

# ------------------------------
# Warm-up Generator using pre-trained M & D with Loss Plot
# ------------------------------
warmup_epochs_MG = 50
batch_size_warmup = 512

feat_layer = mlp.layers[-2]

# for i in range(imbClsNum):
#     dataMinor = trainS[classMap[i]]
#     num_samples = min(2000, dataMinor.shape[0])
#     dataMinor = dataMinor[:num_samples]

#     print(f"Warm-up G guided by M for class {i}, num_samples: {num_samples}")

#     # 儲存每個 epoch 的 Loss
#     loss_history = {'total': [], 'CE': []}

#     for epoch in range(warmup_epochs_MG):
#         idx_shuffle = np.random.permutation(num_samples)
#         dataMinor_shuffled = dataMinor[idx_shuffle]

#         for start in range(0, num_samples, batch_size_warmup):
#             end = min(start + batch_size_warmup, num_samples)
#             batch_real = dataMinor_shuffled[start:end]

#             batch_labels = np.zeros((batch_real.shape[0], c))
#             batch_labels[:, i] = 1

#             batch_real_tf = tf.convert_to_tensor(batch_real, dtype=tf.float32)
#             batch_labels_tf = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

#             z = tf.random.normal((batch_real.shape[0], latDim))

#             with tf.GradientTape() as tape_G_i:
#                 cfmu_features = cfmu_gen([z, batch_labels_tf])
#                 x_fake = gen[i](cfmu_features)

#                 # 用 MLP 計算 cross-entropy loss (frozen)
#                 pred_M = mlp(x_fake, training=False)
#                 ce_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(batch_labels_tf, pred_M))

#             # Update Generator
#             grads = tape_G_i.gradient(ce_loss, gen[i].trainable_variables)
#             optimizer_G[i].apply_gradients(zip(grads, gen[i].trainable_variables))

#         # 儲存 Loss
#         loss_history['total'].append(ce_loss.numpy())
#         loss_history['CE'].append(ce_loss.numpy())

#         # 每個 epoch 印出
#         print(f"Class {i} epoch {epoch+1}/{warmup_epochs_MG}, CE_loss: {ce_loss.numpy():.5f}")

for i in range(imbClsNum):
    dataMinor = trainS[classMap[i]]
    num_samples = min(2000, dataMinor.shape[0])
    dataMinor = dataMinor[:num_samples]

    print(f"Warm-up G guided by M and D for class {i}, num_samples: {num_samples}")

    loss_history = {'total': [], 'O3_O6': [], 'FM': [], 'Recon': []}

    for epoch in range(warmup_epochs_MG):
        idx_shuffle = np.random.permutation(num_samples)
        dataMinor_shuffled = dataMinor[idx_shuffle]

        for start in range(0, num_samples, batch_size_warmup):
            end = min(start + batch_size_warmup, num_samples)
            batch_real = dataMinor_shuffled[start:end]

            batch_labels = np.zeros((batch_real.shape[0], c))
            batch_labels[:, i] = 1

            batch_real_tf = tf.convert_to_tensor(batch_real, dtype=tf.float32)
            batch_labels_tf = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

            z = tf.random.normal((batch_real.shape[0], latDim))

            with tf.GradientTape() as tape_G_i:
                cfmu_features = cfmu_gen([z, batch_labels_tf])
                x_fake = gen[i](cfmu_features)

                # M and D predictions (frozen)
                pred_M = mlp(x_fake, training=False)
                pred_D = dis([x_fake, batch_labels_tf], training=False)

                P_i = P_i_dict[i]
                P_c = P_c_dict[i]
                epsilon = 1e-8

                # O3 / O6
                O3 = (P_c - P_i) * tf.reduce_mean(tf.math.log(pred_M[:, i] + epsilon))
                O6 = P_i * tf.reduce_mean(tf.math.log(1 - pred_D + epsilon))
                G_loss_i = -(O3 - O6)
                O3_dict = {i: O3.numpy()}
                O6_dict = {i: O6.numpy()}

                # Feature Matching Loss
                feat_real = tf.reduce_mean(feat_layer(batch_real_tf), axis=0)
                feat_fake = tf.reduce_mean(feat_layer(x_fake), axis=0)
                fm_loss = tf.reduce_mean(tf.square(feat_real - feat_fake))

                # Reconstruction Loss
                idx = tf.random.uniform([batch_real.shape[0]], maxval=tf.shape(batch_real_tf)[0], dtype=tf.int32)
                real_samples = tf.gather(batch_real_tf, idx)
                recon_loss = tf.reduce_mean(tf.square(x_fake - real_samples))

                # Combine losses
                lambda_fm = 5.0
                lambda_recon = 1.0
                total_loss = G_loss_i + lambda_recon * recon_loss + lambda_fm * fm_loss

            # Update Generator
            grads = tape_G_i.gradient(total_loss, gen[i].trainable_variables)
            optimizer_G[i].apply_gradients(zip(grads, gen[i].trainable_variables))
            P_i_dict = adjust_pi(O3_dict, O6_dict, P_i_dict, target=0, lr=0.05)

        # 儲存 Loss
        loss_history['total'].append(total_loss.numpy())
        loss_history['O3_O6'].append(G_loss_i.numpy())
        loss_history['FM'].append(fm_loss.numpy())
        loss_history['Recon'].append(recon_loss.numpy())

        # ====== 自動評估生成樣本 ======
        # M 分類生成樣本準確率
        y_fake_class = np.argmax(pred_M.numpy(), axis=1)
        acc_fake_class = np.mean(y_fake_class == i)

        # D 被欺騙程度
        avg_D_fake = np.mean(pred_D.numpy())

        print(f"Class {i} epoch {epoch+1}/{warmup_epochs_MG}, "
              f"total_loss: {total_loss.numpy():.5f}, O3_O6: {G_loss_i.numpy():.5f}, "
              f"FM: {fm_loss.numpy():.5f}, Recon: {recon_loss.numpy():.5f}, "
              f"M_acc_fake: {acc_fake_class:.4f}, D_avg_fake: {avg_D_fake:.4f}")

        # 判斷是否可以結束 warm-up
        if acc_fake_class > 0.75 and 0.4 < avg_D_fake < 0.6:
            print(f"Class {i} warm-up condition satisfied at epoch {epoch+1}.")
            break


