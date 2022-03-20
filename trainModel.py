# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/3/15 17:10
# @Function: 模型训练

import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np

dir_name = './'
xTrainPickle = pickle.load(open(dir_name + 'TrainX.pkl', 'rb'))
yTrainPickle = pickle.load(open(dir_name + 'TrainY.pkl', 'rb'))
# xTestPickle = pickle.load(open(dir_name + 'test_x.pkl', 'rb'))
# yTestPickle = pickle.load(open(dir_name + 'test_y.pkl', 'rb'))

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

def testModel():

    xTrain = np.array(xTrainPickle)
    yTrain = np.array(yTrainPickle)
    # xTest = np.array(xTestPickle)
    # yTest = np.array(yTestPickle)

    # xTrain, xTest, max_num = normalization(xTrain, xTest)
    xTrain, xTest, max_num = normalization(xTrain, xTrain)
    yTrain = yTrain / max(yTrain)

    model = RandomForestRegressor(1000)
    model.fit(xTrain, yTrain)
    # scores = model.score(xTest, yTest)
    # yPredicted = model.predict(xTest)
    scores = model.score(xTrain, yTrain)
    yPredicted = model.predict(xTrain)
    i = 0
    t = 0
    yPredicted = sortPredict(yPredicted)

    print(scores)
    print(yPredicted)
    # while i < len(yPredicted):
    #     if abs(round(yPredicted[i]) - yTest[i]) <= 3:
    #         t = t + 1
    #     i = i + 1
    # t = t / len(yPredicted)
    #
    # n = len(yPredicted)
    # rou = 1 - 1 * sum((yPredicted - yTest) ** 2) / (n * (n ** 2 - 1))


def sortPredict(yPredicted):
    y = np.zeros(yPredicted.shape)

    maxValue = max(yPredicted)
    for index in range(len(yPredicted)):
        indexMin = yPredicted.argmin()
        y[indexMin] = index + 1
        yPredicted[indexMin] = maxValue + 1

    return y


# yPredicted = sortPredict(yPredicted)
#
# while i < len(yPredicted):
#     if abs(round(yPredicted[i]) - yTest[i]) <= 3:
#         t = t + 1
#     i = i + 1
# t = t / len(yPredicted)
#
# n = len(yPredicted)
# rou = 1 - 1 * sum((yPredicted - yTest) ** 2) / (n * (n ** 2 - 1))
