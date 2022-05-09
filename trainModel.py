# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/3/15 17:10
# @Function: 模型训练

import pickle

from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Activation, Bidirectional, Embedding, RNN
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

dir_name = './'
xTrainPickle = pickle.load(open(dir_name + 'TrainX.pkl', 'rb'))
yTrainPickle = pickle.load(open(dir_name + 'TrainY.pkl', 'rb'))
xTestPickle = pickle.load(open(dir_name + 'TestX.pkl', 'rb'))
yTestPickle = pickle.load(open(dir_name + 'TestY.pkl', 'rb'))


# 归一化训练集
def normalization(trainData, testData):
    trainDataNormalized = np.zeros(trainData.shape, dtype='float')
    testDataNormalized = np.zeros(testData.shape, dtype='float')

    maxNum = []

    if len(trainData.shape) == 2:
        # 训练集
        for col in range(trainData.shape[1]):
            colTrainData = trainData[..., col]
            colTestData = testData[..., col]

            maxData = max(max(colTrainData), max(colTestData))
            minData = 0
            maxNum.append(maxData)

            colTrainData = (colTrainData - minData) / (maxData - minData + 1)
            colTestData = (colTestData - minData) / (maxData - minData + 1)

            trainDataNormalized[..., col] = colTrainData
            testDataNormalized[..., col] = colTestData

    return trainDataNormalized, testDataNormalized, maxNum


def scale(data):
    # 创建一个缩放器，将数据集中的数据缩放到[-1,1]的取值范围中
    scaler = MinMaxScaler()
    # 使用数据来训练缩放器
    scaler.fit(data)
    # 使用缩放器来将训练集和测试集进行缩放
    data_scaled = scaler.transform(data)
    return scaler, data_scaled

def testModel():
    xTrain = np.array(xTrainPickle)
    yTrain = np.array(yTrainPickle)
    xTest = np.array(xTestPickle)
    yTest = np.array(yTestPickle)

    scaler, xTrain = scale(xTrain)
    scaler, xTest = scale(xTest)
    # xTrain, xTest, max_num = normalization(xTrain, xTest)
    # yTrainMax = max(yTrain)
    # yTrain = yTrain / max(yTrain)

    # 三维化数据，满足LSTM格式
    # xTrainLstm = []
    # for element in xTrain:
    #     tmp = []
    #     for data in element:
    #         dataListed = [data]
    #         tmp.append(dataListed)
    #     xTrainLstm.append(tmp)
    # xTrainLstm = np.array(xTrainLstm)
    # xTestLstm = []
    # for element in xTest:
    #     tmp = []
    #     for data in element:
    #         dataListed = [data]
    #         tmp.append(dataListed)
    #     xTestLstm.append(tmp)
    # xTestLstm = np.array(xTestLstm)

    xTrainLstm = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    xTestLstm = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    model = Sequential()

    model.add(Bidirectional(LSTM(units=50, dropout=0.1, activation='relu')))
    # model.add(LSTM(units=50, dropout=0.1, return_sequences=True, activation='relu'))
    # model.add(LSTM(units=50, dropout=0.1, activation='relu'))

    model.add(Dense(units=1, activation='sigmoid'))

    # optimizer = Adam(learning_rate=0.00015)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(xTrainLstm, yTrain, batch_size=24, epochs=10)
    yPredicted = model.predict(xTestLstm)

    yPredictedListed = []
    for element in yPredicted:
        for value in element:
            yPredictedListed.append(value)
    yPredicted = np.array(yPredictedListed).astype(int)
    print(yPredicted)
    print(yTest)
    print(accuracy_score(yPredicted, yTest))
    print(classification_report(yPredicted, yTest))


def sortPredict(yPredicted):
    y = np.zeros(yPredicted.shape)

    maxValue = max(yPredicted)
    for index in range(len(yPredicted)):
        indexMin = yPredicted.argmin()
        y[indexMin] = index + 1
        yPredicted[indexMin] = maxValue + 1

    return y
