# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from keras.utils import to_categorical

def relabel(labelTr, labelTs):
    unqLab, pInClass = np.unique(labelTr, return_counts=True)
    sortedUnqLab = np.argsort(pInClass, kind='mergesort')
    c = sortedUnqLab.shape[0]
    labelsNewTr = np.zeros((labelTr.shape[0],)) - 1
    labelsNewTs = np.zeros((labelTs.shape[0],)) - 1
    pInClass = np.sort(pInClass)
    classMap = list()
    label_dict = {}  # 加這個

    for i in range(c):
        orig_label = unqLab[sortedUnqLab[i]]
        labelsNewTr[labelTr == orig_label] = i
        labelsNewTs[labelTs == orig_label] = i
        classMap.append(np.where(labelsNewTr == i)[0])
        label_dict[i] = orig_label  # 建立 mapping

    return labelsNewTr, labelsNewTs, c, pInClass, classMap, label_dict

def irFind(pInClass, c, irIgnore=1):
    ir=pInClass[-1]/pInClass
    imbalancedCls=np.arange(c)[ir>irIgnore]
    toBalance=np.subtract(pInClass[-1], pInClass[imbalancedCls])
    imbClsNum=toBalance.shape[0]
    if imbClsNum==0: sys.exit('No imbalanced classes found, exiting ...')
    return imbalancedCls, toBalance, imbClsNum, ir

def fileRead(fileName):
    dataTotal=pd.read_csv(fileName)
    data=dataTotal.iloc[:, :-1].values
    labels=dataTotal.iloc[:, -1].values
    return data, labels

def indices(pLabel, tLabel):
    confMat=confusion_matrix(tLabel, pLabel)
    print(confMat)
    nc=np.sum(confMat, axis=1)
    tp=np.diagonal(confMat)
    tpr=tp/nc
    acsa=np.mean(tpr)
    gm=np.prod(tpr)**(1/confMat.shape[0])
    acc=np.sum(tp)/np.sum(nc)
    return acsa, gm, tpr, confMat, acc

def randomLabelGen(toBalance, batchSize, c):
    cumProb=np.cumsum(toBalance/np.sum(toBalance))
    bins=np.insert(cumProb, 0, 0)
    randomValue=np.random.rand(batchSize,)
    randLabel=np.digitize(randomValue, bins)-1
    randLabel_cat=to_categorical(randLabel)
    labelPadding=np.zeros((batchSize, c-randLabel_cat.shape[1]))
    randLabel_cat=np.hstack((randLabel_cat, labelPadding))
    return randLabel_cat

def batchDivision(n, batchSize):
    numBatches, residual=int(np.ceil(n/batchSize)), int(n%batchSize)
    if residual==0:
        residual=batchSize
    batchDiv=np.zeros((numBatches+1,1), dtype='int64')
    batchSizeStore=np.ones((numBatches, 1), dtype='int64')
    batchSizeStore[0:-1, 0]=batchSize
    batchSizeStore[-1, 0]=residual
    for i in range(numBatches):
        batchDiv[i]=i*batchSize
    batchDiv[numBatches]=batchDiv[numBatches-1]+residual
    return batchDiv, numBatches, batchSizeStore

def rearrange(labelsCat, numImbCls):
    labels=np.argmax(labelsCat, axis=1)
    arrangeMap=list()
    for i in range(numImbCls):
        arrangeMap.append(np.where(labels==i)[0])
    return arrangeMap


def stratified_batchDivision(labels, batchSize, numClasses, min_samples_per_class=1):
    """
    Stratified batch division for imbalanced datasets with oversampling for exhausted classes.

    Args:
        labels: array-like, shape (n,) 整個 dataset 的 label
        batchSize: int, 每個 batch 的大小
        numClasses: int, 類別數
        min_samples_per_class: int, 每個 batch 至少包含的每個類別樣本數

    Returns:
        batchDiv: numpy array, 每個 batch 的起始索引
        numBatches: int, batch 數量
        batchSizeStore: numpy array, 每個 batch 的實際大小
        batch_data_idx: list of arrays, 每個 batch 的索引
    """
    n = len(labels)
    # 將每個 class 的索引存起來
    class_indices = [np.where(labels == i)[0].tolist() for i in range(numClasses)]
    
    # 打亂每個 class 的索引
    for i in range(numClasses):
        np.random.shuffle(class_indices[i])
    
    # 計算 batch 數
    numBatches = int(np.ceil(n / batchSize))
    batchDiv = np.zeros((numBatches + 1,), dtype='int64')
    batchSizeStore = np.zeros((numBatches,), dtype='int64')
    batch_data_idx = []

    current_pos = 0
    for b in range(numBatches):
        batch_idx = []

        # 先從每個 class 抽 min_samples_per_class
        for i in range(numClasses):
            available = class_indices[i]
            if len(available) >= min_samples_per_class:
                take_idx = available[:min_samples_per_class]
                class_indices[i] = available[min_samples_per_class:]
            elif len(available) > 0:
                # 不夠就從已抽過的樣本重採樣
                take_idx = np.random.choice(available, min_samples_per_class, replace=True)
                class_indices[i] = []
            else:
                # 已經沒有剩餘樣本，直接從整個類別重採樣
                orig_idx = np.where(labels == i)[0]
                take_idx = np.random.choice(orig_idx, min_samples_per_class, replace=True)
            batch_idx.extend(take_idx)

        # 再從剩餘索引補足 batchSize
        remaining = batchSize - len(batch_idx)
        all_remaining_idx = np.hstack(class_indices)
        if remaining > 0 and len(all_remaining_idx) > 0:
            take = min(remaining, len(all_remaining_idx))
            batch_idx.extend(all_remaining_idx[:take])
            # 移除已經抽走的索引
            for i in range(numClasses):
                class_indices[i] = list(set(class_indices[i]) - set(all_remaining_idx[:take]))

        batch_idx = np.array(batch_idx, dtype=int)
        batchSizeStore[b] = len(batch_idx)
        batchDiv[b] = current_pos
        current_pos += len(batch_idx)
        batch_data_idx.append(batch_idx)

    batchDiv[numBatches] = current_pos
    return batchDiv, numBatches, batchSizeStore, batch_data_idx








