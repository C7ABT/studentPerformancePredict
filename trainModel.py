# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/3/15 17:10
# @Function: 模型训练

import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Activation
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

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


def scale(trainData, testData):
    # 创建一个缩放器，将数据集中的数据缩放到[-1,1]的取值范围中
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 使用数据来训练缩放器
    scaler.fit(trainData)
    # 使用缩放器来将训练集和测试集进行缩放
    trainDataScaled = scaler.transform(trainData)
    testDataScaled = scaler.transform(testData)
    return scaler, trainDataScaled, testDataScaled


# 将预测值进行逆缩放，使用之前训练好的缩放器，x为一维数组，y为实数
def invert_scale(scaler, X, y):
    # 将X,y转换为一个list列表
    new_row = [x for x in X] + [y]
    # 将列表转换为数组
    array = np.array(new_row)
    # 将数组重构成一个形状为[1,2]的二维数组->[[10,12]]
    array = array.reshape(1, len(array))
    # 逆缩放输入的形状为[1,2]，输出形状也是如此
    invert = scaler.inverse_transform(array)
    # 只需要返回y值即可
    return invert[0, -1]


def testModel():
    # xTrain: (430, 278)
    # yTrain: (430)
    xTrain = np.array(xTrainPickle)
    yTrain = np.array(yTrainPickle)
    xTest = np.array(xTestPickle)
    yTest = np.array(yTestPickle)

    # scalerXTrain, xTrain = scale(xTrain)
    # scalerXTest, xTest = scale(xTest)
    # scaler, xTrain, xTest = scale(xTrain, xTest)
    # scalerYTrain, yTrain = scale(yTrain)
    scalerYTest, yTest = scale(yTest)
    # xTrain, xTest, max_num = normalization(xTrain, xTest)
    yTrain = yTrain / max(yTrain)

    # 三维化数据，满足LSTM格式
    xTrainLstm = []
    for element in xTrain:
        tmp = []
        for data in element:
            dataListed = [data]
            tmp.append(dataListed)
        xTrainLstm.append(tmp)
    xTrainLstm = np.array(xTrainLstm)
    xTestLstm = []
    for element in xTest:
        tmp = []
        for data in element:
            dataListed = [data]
            tmp.append(dataListed)
        xTestLstm.append(tmp)
    xTestLstm = np.array(xTestLstm)
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(None, 1), return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(xTrainLstm, yTrain, batch_size=32, epochs=5, validation_split=0.1)
    yPredicted = model.predict(xTestLstm)
    print(yPredicted)
    print(yTest)

    # model = RandomForestRegressor(1000)
    # model.fit(xTrain, yTrain)
    # yPredicted = model.predict(xTest)
    # # yPredicted = sortPredict(yPredicted)
    # yPredicted = yPredicted * 538
    # print(yPredicted.astype(np.int))
    # print(yTest)

    # n = len(yPredicted)
    # rou = 1 - 6 * sum((yPredicted - yTest) ** 2) / (n * (n ** 2 - 1))
    # print(rou)


def sortPredict(yPredicted):
    y = np.zeros(yPredicted.shape)

    maxValue = max(yPredicted)
    for index in range(len(yPredicted)):
        indexMin = yPredicted.argmin()
        y[indexMin] = index + 1
        yPredicted[indexMin] = maxValue + 1

    return y
