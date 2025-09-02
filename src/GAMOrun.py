# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import os
import numpy as np
import dense_suppli as spp
import dense_net as nt
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import tensorflow as tf
import json

# For selecting a GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_optimizer():
    return Adam(0.0002, 0.5)

# 確保啟用 eager execution
tf.config.run_functions_eagerly(True)

# Ground works
fileName=['./extra_dataset/df_minority_train.csv', './extra_dataset/df_minority_test.csv']
fileStart='UBSW_NB15_Gamo'
fileEnd, savePath='_Model.h5', fileStart+'/'
latDim, modelSamplePd, resSamplePd=32, 250, 50

batchSize, max_step=256, 500

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
    optimizer_G.append(Adam(0.0002, 0.5))
optimizer_D = Adam(0.0002, 0.5)
optimizer_M = Adam(0.0002, 0.5)

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

cfmu_gen=nt.build_cfmu()
cfmu_gen.compile(loss='mean_squared_error', optimizer=get_optimizer())

gen, genP_mlp, genP_dis=list(), list(), list()
for i in range(imbClsNum):
    dataMinor=trainS[classMap[i], :]
    numMinor=dataMinor.shape[0]
    gen.append(nt.denseGamoGenCreate(latDim, numMinor, dataMinor))
    gen[i].compile(optimizer=optimizer_G[i])

    ip1=Input(shape=(latDim,))
    ip2=Input(shape=(c,))
    op1=cfmu_gen([ip1, ip2])
    op2=gen[i](op1)
    op3=mlp(op2)
    genP_mlp.append(Model(inputs=op1, outputs=op3))
    genP_mlp[i].compile(loss='mean_squared_error', optimizer=get_optimizer())

    ip1=Input(shape=(latDim,))
    ip2=Input(shape=(c,))
    ip3=Input(shape=(c,))
    op1=cfmu_gen([ip1, ip2])
    op2=gen[i](op1)
    op3=dis([op2, ip3])
    genP_dis.append(Model(inputs=[op1, ip3], outputs=op3))
    genP_dis[i].compile(loss='mean_squared_error', optimizer=get_optimizer())

print(pInClass)

num_classes = len(np.unique(labelTr))
P_i_dict = {}
P_i_dict = {i: (1.0/N_i) / sum(1.0/np.array(pInClass)) for i, N_i in enumerate(pInClass)}
P_i_sum = sum(P_i_dict.values())
P_i_dict = {i: P_i_dict[i] / P_i_sum for i in range(c)}  # normalize to sum 1
P_c_dict = {i: 1.0 for i in range(c)}

# for record saving
# ---------------- Stratified Batch Division ----------------
batchDiv, numBatches, bSStore, batch_data_idx = spp.stratified_batchDivision(
    labels=labelTr, 
    batchSize=batchSize, 
    numClasses=c, 
    min_samples_per_class=3
)

genClassPoints = int(np.ceil(batchSize / c))

if not os.path.exists(fileStart):
    os.makedirs(fileStart)

iter=int(np.ceil(max_step/resSamplePd)+1)
acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
confMatSaveTr, confMatSaveTs=np.zeros((iter, c, c)), np.zeros((iter, c, c))
tprSaveTr, tprSaveTs=np.zeros((iter, c)), np.zeros((iter, c))

# training

print(classMap)

noise_dim = latDim
num_classes = c

step = 0
while step < max_step:
    for j in range(numBatches):
        idx_batch = batch_data_idx[j]  # 使用 stratified batch 索引
        batch_real = trainS[idx_batch]
        batch_labels = labelsCat[idx_batch]

        batch_real_tf = tf.convert_to_tensor(batch_real, dtype=tf.float32)
        batch_labels_tf = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

        # ------------------------------
        # 1️⃣ 更新 Discriminator
        # ------------------------------
        grads_list_M = []
        for i in range(imbClsNum):
            class_idx = np.where(np.argmax(batch_labels, axis=1) == i)[0]
            if len(class_idx) == 0:
                continue

            dataMinor_batch = tf.convert_to_tensor(batch_real[class_idx], dtype=tf.float32)
            labels_onehot = tf.convert_to_tensor(batch_labels[class_idx], dtype=tf.float32)

            z = tf.random.normal((tf.shape(dataMinor_batch)[0], noise_dim))
            cfmu_features = cfmu_gen([z, labels_onehot], training=True)
            x_fake = gen[i](cfmu_features)

            with tf.GradientTape() as tape_M_i:
                pred_real = mlp(dataMinor_batch, training=True)
                pred_fake = mlp(x_fake, training=True)

                # 直接用 categorical crossentropy
                all_labels = tf.concat([labels_onehot, labels_onehot], axis=0)  # 生成樣本的 label same as minority class
                all_preds = tf.concat([pred_real, pred_fake], axis=0)
                M_loss_i = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(all_labels, all_preds))

                # M_loss_i = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels_onehot, pred_real))

            grads_M_i = tape_M_i.gradient(M_loss_i, mlp.trainable_variables)
            if grads_M_i is not None:
                grads_list_M.append(grads_M_i)

            # with tf.GradientTape() as tape_M_i:
            #     pred_real = mlp(dataMinor_batch, training=True)
            #     pred_fake = mlp(x_fake, training=True)

            #     # print(pred_real)

            #     P_i = P_i_dict[i]
            #     P_c = P_c_dict[i]

            #     # print(pred_real)

            #     O1 = P_i * tf.reduce_mean(tf.math.log(pred_real[:, i] + 1e-8))
            #     O2 = tf.reduce_mean(tf.math.log(1 - pred_real + 1e-8))
            #     O3 = (P_c - P_i) * tf.reduce_mean(tf.math.log(pred_fake[:, i] + 1e-8))
            #     O4 = tf.reduce_mean(tf.math.log(1 - pred_fake + 1e-8))

            #     M_loss_i = -(O1 - O2 + O3 - O4)

            # grads_M_i = tape_M_i.gradient(M_loss_i, mlp.trainable_variables)
            # if grads_M_i is not None:
            #     grads_list_M.append(grads_M_i)
            

            with tf.GradientTape() as tape_D_i:

                D_real = dis([dataMinor_batch, labels_onehot], training=True)
                D_fake = dis([x_fake, labels_onehot], training=True)

                P_i = P_i_dict[i]
                O5 = P_i * tf.math.log(D_real + 1e-8)
                O6 = P_i * tf.math.log(1 - D_fake + 1e-8)

                D_loss_i = -tf.reduce_mean(O5 + O6)

            grads_i = tape_D_i.gradient(D_loss_i, dis.trainable_variables)
            if grads_i is not None:
                optimizer_D.apply_gradients(zip(grads_i, dis.trainable_variables))
        
        if grads_list_M:
            avg_grads_M = [
                tf.reduce_mean(tf.stack([g[i] for g in grads_list_M]), axis=0)
                for i in range(len(mlp.trainable_variables))
            ]
        optimizer_M.apply_gradients(zip(avg_grads_M, mlp.trainable_variables))

        # ------------------------------
        # 3️⃣ 更新 Generator
        # ------------------------------
        for i in range(imbClsNum):

            # 確保每個少數類別都有樣本
            gen_batch_size = max(1, np.sum(np.argmax(batch_labels, axis=1) == i))
            z = tf.random.normal((gen_batch_size, noise_dim))
            label_onehot = np.zeros((gen_batch_size, c))
            label_onehot[:, i] = 1
            label_onehot_tf = tf.convert_to_tensor(label_onehot, dtype=tf.float32)

            with tf.GradientTape() as tape_G_i:
                cfmu_features = cfmu_gen([z, label_onehot_tf], training=True)
                x_fake = gen[i](cfmu_features)

                pred_M = mlp(x_fake)
                pred_D = dis([x_fake, label_onehot_tf])

                P_i = P_i_dict[i]
                P_c = P_c_dict[i]

                # 依據你的公式更新 O3、O6
                O3 = (P_c - P_i) * tf.reduce_mean(tf.math.log(pred_M[:, i] + 1e-8))
                O6 = P_i * tf.reduce_mean(tf.math.log(1 - pred_D + 1e-8))

                G_loss_i = -(O3 - O6)

            grads_G_i = tape_G_i.gradient(G_loss_i, gen[i].trainable_variables)
            if grads_G_i is not None:
                optimizer_G[i].apply_gradients(zip(grads_G_i, gen[i].trainable_variables))

        if step%resSamplePd==0:
            saveStep=int(step//resSamplePd)

            pLabel=np.argmax(mlp.predict(trainS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
            print('Train: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTr[saveStep], gmSaveTr[saveStep], accSaveTr[saveStep]=acsa, gm, acc
            confMatSaveTr[saveStep]=confMat
            tprSaveTr[saveStep]=tpr

            pLabel=np.argmax(mlp.predict(testS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
            print('Test: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTs[saveStep], gmSaveTs[saveStep], accSaveTs[saveStep]=acsa, gm, acc
            confMatSaveTs[saveStep]=confMat
            tprSaveTs[saveStep]=tpr

        if step%modelSamplePd==0 and step!=0:
            direcPath=savePath+'gamo_models_'+str(step)
            if not os.path.exists(direcPath):
                os.makedirs(direcPath)
            mlp.save(direcPath+'/MLP_'+str(step)+fileEnd)
            dis.save(direcPath+'/DIS_'+str(step)+fileEnd)
            for i in range(imbClsNum):
                gen[i].save(direcPath+'/GenForClass_'+str(i)+'_'+str(step)+fileEnd)

        step += 2
        # if step % resSamplePd == 0:
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