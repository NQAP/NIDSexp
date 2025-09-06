# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import os
import numpy as np
import dense_suppli as spp
import dense_net as nt
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
import tensorflow as tf
import json

# For selecting a GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# 確保啟用 eager execution
tf.config.run_functions_eagerly(True)

# Ground works
fileName=['./extra_dataset/df_minority_train.csv', './extra_dataset/df_minority_test.csv']
fileStart='Mnist_100_Gamo'
fileEnd, savePath='_Model.h5', fileStart+'/'
def get_optimizer():
    return Adam(0.0002, 0.5)
latDim, modelSamplePd, resSamplePd=100, 5000, 500
plt.ion()
adamOpt=Adam(0.0002, 0.5)

batchSize, max_step=32, 50000

trainS, labelTr=spp.fileRead(fileName[0])
testS, labelTs=spp.fileRead(fileName[1])

n, m=trainS.shape[0], testS.shape[0]
trainS, testS=(trainS-127.5)/127.5, (testS-127.5)/127.5

labelTr, labelTs, c, pInClass, classMap, label_dict=spp.relabel(labelTr, labelTs)
imbalancedCls, toBalance, imbClsNum, ir=spp.irFind(pInClass, c)

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
mlp.compile(loss='mean_squared_error', optimizer=adamOpt)
mlp.trainable=False

dis=nt.denseDisCreate()
dis.compile(loss='mean_squared_error', optimizer=adamOpt)
dis.trainable=False

# Ground works
fileName=['./extra_dataset/df_minority_train.csv', './extra_dataset/df_minority_test.csv']
fileStart='UBSW_NB15_Gamo_New'
fileEnd, savePath='_Model.h5', fileStart+'/'
latDim, modelSamplePd, resSamplePd=32, 500, 100

batchSize, max_step=32, 5000

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
    
gen_processed, genP_mlp, genP_dis=list(), list(), list()
for i in range(imbClsNum):
    dataMinor=trainS[classMap[i], :]
    numMinor=dataMinor.shape[0]
    gen.append(nt.denseGamoGenCreate(latDim, numMinor, dataMinor))

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

# for record saving
batchDiv, numBatches, bSStore=spp.batchDivision(n, batchSize)
genClassPoints=int(np.ceil(batchSize/c))
fig, axs=plt.subplots(imbClsNum, 3)

if not os.path.exists(fileStart):
    os.makedirs(fileStart)


iter=int(np.ceil(max_step/resSamplePd)+1)
acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
confMatSaveTr, confMatSaveTs=np.zeros((iter, c, c)), np.zeros((iter, c, c))
tprSaveTr, tprSaveTs=np.zeros((iter, c)), np.zeros((iter, c))

# training
step=0
while step<max_step:
    for j in range(numBatches):
        x1, x2=batchDiv[j, 0], batchDiv[j+1, 0]
        validR=np.ones((bSStore[j, 0],1))-np.random.uniform(0,0.1, size=(bSStore[j, 0], 1))
        mlp.train_on_batch(trainS[x1:x2], labelsCat[x1:x2])
        dis.train_on_batch([trainS[x1:x2], labelsCat[x1:x2]], validR)

        invalid=np.zeros((bSStore[j, 0], 1))+np.random.uniform(0, 0.1, size=(bSStore[j, 0], 1))
        randNoise=np.random.normal(0, 1, (bSStore[j, 0], latDim))
        fakeLabel=spp.randomLabelGen(toBalance, bSStore[j, 0], c)
        rLPerClass=spp.rearrange(fakeLabel, imbClsNum)
        fakePoints=np.zeros((bSStore[j, 0], 42))
        cfmu = cfmu_gen([randNoise, fakeLabel])
        for i1 in range(imbClsNum):
            if rLPerClass[i1].shape[0]!=0:
                gen_i=gen[i1].predict(cfmu)
                # print(gen_i.shape)
                fakePoints[rLPerClass[i1]]=gen_i[rLPerClass[i1]]

        mlp.train_on_batch(fakePoints, fakeLabel)
        dis.train_on_batch([fakePoints, fakeLabel], invalid)

        for i1 in range(imbClsNum):
            validA=np.ones((genClassPoints, 1))
            randomLabel=np.zeros((genClassPoints, c))
            randomLabel[:, i1]=1
            randNoise=np.random.normal(0, 1, (genClassPoints, latDim))
            oppositeLabel=np.ones((genClassPoints, c))-randomLabel
            cfmu = cfmu_gen([randNoise, randomLabel])
            genP_mlp[i1].train_on_batch(cfmu, oppositeLabel)
            genP_dis[i1].train_on_batch([cfmu, tf.constant(randomLabel, dtype=tf.float32)], (validA))

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
                gen[i].save(direcPath+'/GenP_'+str(i)+'_'+str(step)+fileEnd)

        step=step+2
        if step>=max_step: break

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
    gen[i].save(direcPath+'/GenP_'+str(i)+'_'+str(step)+fileEnd)

resSave=savePath+'Results'
np.savez(resSave, acsa=acsa, gm=gm, tpr=tpr, confMat=confMat, acc=acc)
recordSave=savePath+'Record'
np.savez(recordSave, acsaSaveTr=acsaSaveTr, gmSaveTr=gmSaveTr, accSaveTr=accSaveTr, acsaSaveTs=acsaSaveTs, gmSaveTs=gmSaveTs, accSaveTs=accSaveTs, confMatSaveTr=confMatSaveTr, confMatSaveTs=confMatSaveTs, tprSaveTr=tprSaveTr, tprSaveTs=tprSaveTs)