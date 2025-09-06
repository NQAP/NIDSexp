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
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle

def load_models(gen_prefix="./UBSW_NB15_Gamo_New/gamo_models_"):
    """
    載入所有生成器和 cfmu_gen
    """
    gen = []
    mlp_path = f"{gen_prefix}/MLP__Model.h5"
    dis_path = f"{gen_prefix}/DIS__Model.h5"
    print(f"載入生成器: {mlp_path}")
    mlp = load_model(mlp_path, compile=False, custom_objects={"SelfAttention": nt.SelfAttention}, safe_mode=False)
    dis = load_model(dis_path, compile=False, custom_objects={"SelfAttention": nt.SelfAttention}, safe_mode=False)
    # for i in range(7):
    #     gen_path = f"{gen_prefix}/GenForClass_{i}__Model.h5"
    #     gen.append(load_model(gen_path, compile=False, custom_objects={"SelfAttention": nt.SelfAttention, "GenProcessFinal": nt.GenProcessFinal}, safe_mode=False))
    return mlp, dis

# For selecting a GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_optimizer(lr=5e-4):
    return Adam(lr, 0.5)

# 確保啟用 eager execution
tf.config.run_functions_eagerly(True)

# Ground works
fileName=['./extra_dataset/df_minority_train.csv', './extra_dataset/df_minority_test.csv']
fileStart='UBSW_NB15_Gamo_Ver2'
fileEnd, savePath='_Model.h5', fileStart+'/'
latDim, modelSamplePd, resSamplePd=32, 500, 50

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
    optimizer_G.append(Adam(1e-4, 0.5))

labelsCat=to_categorical(labelTr)

shuffleIndex=np.random.choice(np.arange(n), size=(n,), replace=False)
trainS=trainS[shuffleIndex]
labelTr=labelTr[shuffleIndex]
labelsCat=labelsCat[shuffleIndex]
classMap=list()
for i in range(c):
    classMap.append(np.where(labelTr==i)[0])

mlp, dis = load_models()

mlp.trainable = False
# freeze D
dis.trainable = False

cfmu_gen = load_model(
    "./model/cfmu_model/pretrained_cfmu_label_guided.h5", 
    custom_objects={"SelfAttention": nt.SelfAttention}
)

cfmu_gen.trainable = False  # 鎖定權重

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
alpha = 0.5  # 平滑指數，避免極端
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

noise_dim = latDim
num_classes = c

optimizer_G = []
for i in range(imbClsNum):
    optimizer_G.append(Adam(2e-4, 0.5))

gen =[]
for i in range(imbClsNum):
    dataMinor=trainS[classMap[i], :]
    numMinor=dataMinor.shape[0]
    gen.append(nt.denseGamoGenCreate(latDim, numMinor, dataMinor))
    gen[i].compile(optimizer=optimizer_G[i])

# ------------------------------
# Loss 儲存 (每個少數類別分開)
# ------------------------------
num_iter = int(np.ceil(max_step / resSamplePd)) + 1
M_loss_save = np.zeros((num_iter,))
D_loss_save = np.zeros((num_iter, imbClsNum))
G_loss_save = np.zeros((num_iter, imbClsNum))

# ------------------------------
# Warm-up Generator using pre-trained M & D with Loss Plot
# ------------------------------
warmup_epochs_MG = 20
batch_size_warmup = 512

feat_layer = mlp.layers[-2]

for i in range(imbClsNum):
    dataMinor = trainS[classMap[i]]
    num_samples = min(2500, dataMinor.shape[0])
    dataMinor = dataMinor[:num_samples]

    print(f"Warm-up G guided by M and D for class {i}, num_samples: {num_samples}")

    # 儲存每個 epoch 的 Loss
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
                # 計算 O3/O6 後，更新 P_i
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

        # 每個 epoch 印出
        print(f"Class {i} epoch {epoch+1}/{warmup_epochs_MG}, total_loss: {total_loss.numpy():.5f}, "
              f"O3_O6: {G_loss_i.numpy():.5f}, FM: {fm_loss.numpy():.5f}, Recon: {recon_loss.numpy():.5f}")
        
direcPath = savePath + 'gamo_models_'
for i in range(imbClsNum):
    gen[i].save(direcPath + '/GenForClass_' + str(i) + '_'  + fileEnd)

optimizer_D = get_optimizer(1e-5)
optimizer_M = get_optimizer(5e-4)
for i in range(imbClsNum):
    optimizer_G.append(get_optimizer(1e-4))

step = 0
mlp.trainable = True
dis.trainable = True
while step < max_step:
    for j in range(numBatches):
        idx_batch = batch_data_idx[j]  # stratified batch index
        batch_real = trainS[idx_batch]
        batch_labels = labelsCat[idx_batch]

        batch_real_tf = tf.convert_to_tensor(batch_real, dtype=tf.float32)
        batch_labels_tf = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

        # ------------------------------
        # 更新 Classifier M
        # ------------------------------
        x_fake_list, y_fake_list, weight_fake_list = [], [], []

        for i in range(imbClsNum):
            gen_batch_size = max(1, np.sum(np.argmax(batch_labels, axis=1) == i))
            z = tf.random.normal((gen_batch_size, noise_dim))
            label_onehot = np.zeros((gen_batch_size, c))
            label_onehot[:, i] = 1
            label_onehot_tf = tf.convert_to_tensor(label_onehot, dtype=tf.float32)

            cfmu_features = cfmu_gen([z, label_onehot_tf])
            x_fake = gen[i](cfmu_features)
            pred_D = dis([x_fake, label_onehot_tf], training=False)

            weights = tf.squeeze(pred_D, axis=-1)
            x_fake_list.append(x_fake)
            y_fake_list.append(label_onehot_tf)
            weight_fake_list.append(weights)

        # 合併所有 fake sample
        x_fake_all = tf.concat(x_fake_list, axis=0)
        y_fake_all = tf.concat(y_fake_list, axis=0)
        weights_fake_all = tf.concat(weight_fake_list, axis=0)

        # 合併 real + fake
        x_m_train = tf.concat([batch_real_tf, x_fake_all], axis=0)
        y_m_train = tf.concat([batch_labels_tf, y_fake_all], axis=0)
        weights_m = tf.concat([tf.ones(batch_real_tf.shape[0]), weights_fake_all], axis=0)

        with tf.GradientTape() as tape_M:
            preds = mlp(x_m_train, training=True)

            # 只用 tf.keras.losses.categorical_crossentropy，並加權
            pred_real = preds[:batch_real_tf.shape[0]]
            pred_fake = preds[batch_real_tf.shape[0]:]

            # 原始樣本
            loss_real = tf.keras.losses.categorical_crossentropy(batch_labels_tf, pred_real)
            loss_real = tf.reduce_mean(loss_real)  # 平均

            # 生成樣本（加權）
            loss_fake = tf.keras.losses.categorical_crossentropy(y_fake_all, pred_fake)
            loss_fake = tf.reduce_mean(loss_fake * weights_fake_all * class_weights_tf[i])

            # 總 loss
            λ_M_real = 1.0
            λ_M_fake = 0.8
            loss_M = λ_M_real * loss_real + λ_M_fake * loss_fake


        grads_M = tape_M.gradient(loss_M, mlp.trainable_variables)
        optimizer_M.apply_gradients(zip(grads_M, mlp.trainable_variables))


        # ------------------------------
        # 1️⃣ 更新 Discriminator (D)
        # ------------------------------
        D_loss_classes = []
        with tf.GradientTape() as tape_D:
            total_loss = 0
            count = 0
            for i in range(imbClsNum):
                class_idx = np.where(np.argmax(batch_labels, axis=1) == i)[0]
                if len(class_idx) == 0:
                    D_loss_classes.append(0.0)
                    continue

                dataMinor_batch = tf.convert_to_tensor(batch_real[class_idx], dtype=tf.float32)
                labels_onehot = tf.convert_to_tensor(batch_labels[class_idx], dtype=tf.float32)

                z = tf.random.normal((tf.shape(dataMinor_batch)[0], noise_dim))
                cfmu_features = cfmu_gen([z, labels_onehot])
                x_fake = gen[i](cfmu_features)

                D_real = dis([dataMinor_batch, labels_onehot], training=True)
                D_fake = dis([x_fake, labels_onehot], training=True)

                P_i = P_i_dict[i]
                O5 = P_i * tf.math.log(D_real + 1e-8)
                O6 = P_i * tf.math.log(1 - D_fake + 1e-8)
                D_loss_i = - class_weights_tf[i] * tf.reduce_mean(O5 + O6)
                total_loss += D_loss_i 
                count += 1
                D_loss_classes.append(D_loss_i.numpy())
                

            D_loss_avg = total_loss / max(1, count)
            λ_D = 1.0
            D_loss_avg *= λ_D

        grads = tape_D.gradient(D_loss_avg, dis.trainable_variables)
        optimizer_D.apply_gradients(zip(grads, dis.trainable_variables))


        # ------------------------------
        # 2️⃣ 更新 Generator (G)
        # ------------------------------
        G_loss_classes = []
        for i in range(imbClsNum):
            gen_batch_size = max(1, np.sum(np.argmax(batch_labels, axis=1) == i))
            z = tf.random.normal((gen_batch_size, noise_dim))
            label_onehot = np.zeros((gen_batch_size, c))
            label_onehot[:, i] = 1
            label_onehot_tf = tf.convert_to_tensor(label_onehot, dtype=tf.float32)

            with tf.GradientTape() as tape_G_i:
                cfmu_features = cfmu_gen([z, label_onehot_tf])
                x_fake = gen[i](cfmu_features)

                pred_M = mlp(x_fake, training=False)
                pred_D = dis([x_fake, label_onehot_tf], training=False)

                # Generator 欺騙 D
                G_loss_D = -tf.reduce_mean(tf.math.log(pred_D + 1e-8))

                # Generator 讓 M 對該 class 得分最大
                G_loss_M_class = -tf.reduce_mean(tf.math.log(pred_M[:, i] + 1e-8))

                # Feature Matching
                feat_real = tf.reduce_mean(mlp.layers[-2](batch_real_tf), axis=0)
                feat_fake = tf.reduce_mean(mlp.layers[-2](x_fake), axis=0)
                fm_loss = tf.reduce_mean(tf.square(feat_real - feat_fake))

                # Reconstruction
                idx = tf.random.uniform([gen_batch_size], maxval=tf.shape(batch_real_tf)[0], dtype=tf.int32)
                real_samples = tf.gather(batch_real_tf, idx)
                recon_loss = tf.reduce_mean(tf.square(x_fake - real_samples))

                # 權重
                λ_G_D = 1.0
                λ_G_M_class = 1.0
                λ_fm = 1.0
                λ_recon = 0.5

                total_loss = λ_G_D * G_loss_D + λ_G_M_class * G_loss_M_class + λ_fm * fm_loss + λ_recon * recon_loss

            grads_G_i = tape_G_i.gradient(total_loss, gen[i].trainable_variables)
            optimizer_G[i].apply_gradients(zip(grads_G_i, gen[i].trainable_variables))
            G_loss_classes.append(total_loss.numpy())


        # ------------------------------
        # 3️⃣ Logging + Save (M 不更新)
        # ------------------------------
        if step % resSamplePd == 0:
            saveStep = int(step // resSamplePd)
            direcPath = savePath + 'gamo_models_' + str(step)
            os.makedirs(direcPath, exist_ok=True)

            # 只計算 M 對 train / test 的效果，不更新 M
            pLabel=np.argmax(mlp.predict(trainS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
            print('Performance on Train Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTr[-1], gmSaveTr[-1], accSaveTr[-1]=acsa, gm, acc
            confMatSaveTr[-1]=confMat
            tprSaveTr[-1]=tpr

            pLabel=np.argmax(mlp.predict(testS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
            print('Performance on Test Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTs[-1], gmSaveTs[-1], accSaveTs[-1]=acsa, gm, acc
            confMatSaveTs[-1]=confMat
            tprSaveTs[-1]=tpr

            metrics = {
                'acsaSaveTr': acsaSaveTr,
                'gmSaveTr': gmSaveTr,
                'accSaveTr': accSaveTr,
                'confMatSaveTr': confMatSaveTr,
                'tprSaveTr': tprSaveTr,
                'acsaSaveTs': acsaSaveTs,
                'gmSaveTs': gmSaveTs,
                'accSaveTs': accSaveTs,
                'confMatSaveTs': confMatSaveTs,
                'tprSaveTs': tprSaveTs
            }
            with open(direcPath + '/metrics_'+ str(step) +'.pkl', 'wb') as f:
                pickle.dump(metrics, f)

            # 儲存 loss
            M_loss_save[saveStep] = loss_M  # M 固定不更新
            D_loss_save[saveStep, :] = np.array(D_loss_classes)
            G_loss_save[saveStep, :] = np.array(G_loss_classes)

            print(f"Step {step}: M_loss={M_loss_save[saveStep]}, D_loss={D_loss_save[saveStep]}, G_loss={G_loss_save[saveStep]}")

        # ------------------------------
        # 4️⃣ 模型存檔
        # ------------------------------
        if step % modelSamplePd == 0 and step != 0:
            direcPath = savePath + 'gamo_models_' + str(step)
            os.makedirs(direcPath, exist_ok=True)
            mlp.save(direcPath + '/MLP_' + str(step) + fileEnd)
            dis.save(direcPath + '/DIS_' + str(step) + fileEnd)
            for i in range(imbClsNum):
                gen[i].save(direcPath + '/GenForClass_' + str(i) + '_' + str(step) + fileEnd)

        step += 1
        print(f"Step {step} finished.")
        if step >= max_step:
            break

pLabel=np.argmax(mlp.predict(trainS), axis=1)
acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
print('Performance on Train Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
print('TPR: ', np.round(tpr, 2))
acsaSaveTr[-1], gmSaveTr[-1], accSaveTr[-1]=acsa, gm, acc
confMatSaveTr[-1]=confMat
tprSaveTr[-1]=tpr

pLabel=np.argmax(mlp.predict(testS), axis=1)
acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
print('Performance on Test Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
print('TPR: ', np.round(tpr, 2))
acsaSaveTs[-1], gmSaveTs[-1], accSaveTs[-1]=acsa, gm, acc
confMatSaveTs[-1]=confMat
tprSaveTs[-1]=tpr

direcPath=savePath+'gamo_models_'+str(step)
if not os.path.exists(direcPath):
    os.makedirs(direcPath)
mlp.save(direcPath+'/MLP_'+str(step)+fileEnd)
dis.save(direcPath+'/DIS_'+str(step)+fileEnd)
for i in range(imbClsNum):
    gen[i].save(direcPath+'/GenForClass_'+str(i)+'_'+str(step)+fileEnd)

resSave=savePath+'Results'
np.savez(resSave, acsa=acsa, gm=gm, tpr=tpr, confMat=confMat, acc=acc)
recordSave=savePath+'Record'
np.savez(recordSave, acsaSaveTr=acsaSaveTr, gmSaveTr=gmSaveTr, accSaveTr=accSaveTr, acsaSaveTs=acsaSaveTs, gmSaveTs=gmSaveTs, accSaveTs=accSaveTs, confMatSaveTr=confMatSaveTr, confMatSaveTs=confMatSaveTs, tprSaveTr=tprSaveTr, tprSaveTs=tprSaveTs)

# ------------------------------
# Loss 畫圖 (每個少數類別的 G_loss 與 D_loss + M_loss)
# ------------------------------
x_axis = np.arange(0, max_step + 1, resSamplePd)

# 統一 dtype
M_loss_plot = M_loss_save.astype(np.float32)
D_loss_plot = D_loss_save.astype(np.float32)
G_loss_plot = G_loss_save.astype(np.float32)

plt.figure(figsize=(14, 7))

# M_loss
plt.plot(x_axis, M_loss_plot, label='M_loss', color='black', linewidth=2)

colors = plt.cm.tab10.colors  # 前10個顏色

# D_loss (虛線) & G_loss (實線)
for i in range(imbClsNum):
    plt.plot(x_axis, G_loss_plot[:, i], label=f'G_loss class {i}', color=colors[i % 10], linestyle='-')
    plt.plot(x_axis, D_loss_plot[:, i], label=f'D_loss class {i}', color=colors[i % 10], linestyle='--')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('MLP, Generator & Discriminator Loss Curves (per class)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()