# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/3/15 17:10
# @Function: 模型训练

import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import numpy as np

dir_name = './'
x1TrainPickle = pickle.load(open(dir_name + 'TrainX1.pkl', 'rb'))
y1TrainPickle = pickle.load(open(dir_name + 'TrainY1.pkl', 'rb'))
x2TrainPickle = pickle.load(open(dir_name + 'TrainX2.pkl', 'rb'))
y2TrainPickle = pickle.load(open(dir_name + 'TrainY2.pkl', 'rb'))
xTestPickle = pickle.load(open(dir_name + 'TestX.pkl', 'rb'))
yTestPickle = pickle.load(open(dir_name + 'TestY.pkl', 'rb'))

xTrainPickle = pickle.load(open(dir_name + 'TrainX.pkl', 'rb'))
yTrainPickle = pickle.load(open(dir_name + 'TrainY.pkl', 'rb'))

# 归一化数据集
def normalization(trainData1, trainData2, testData):
    trainData1Normalized = np.zeros(trainData1.shape, dtype='float')
    trainData2Normalized = np.zeros(trainData2.shape, dtype='float')
    testDataNormalized = np.zeros(testData.shape, dtype='float')

    for col in range(trainData1.shape[1]):
        colTrainData1 = trainData1[..., col]
        colTrainData2 = trainData2[..., col]
        colTestData = testData[..., col]

        maxData = max(max(colTrainData1), max(colTrainData2), max(colTestData))
        minData = 0

        colTrainData1 = (colTrainData1 - minData) / (maxData - minData + 1)
        colTrainData2 = (colTrainData2 - minData) / (maxData - minData + 1)
        colTestData = (colTestData - minData) / (maxData - minData + 1)

        trainData1Normalized[..., col] = colTrainData1
        trainData2Normalized[..., col] = colTrainData2
        testDataNormalized[..., col] = colTestData

    return trainData1Normalized, trainData2Normalized, testDataNormalized


def testModel():
    xTrain1 = np.array(x1TrainPickle)
    yTrain1 = np.array(y1TrainPickle)
    xTrain2 = np.array(x2TrainPickle)
    yTrain2 = np.array(y2TrainPickle)
    xTest = np.array(xTestPickle)
    yTest = np.array(yTestPickle)

    xTrain1, xTrain2, xTest = normalization(xTrain1, xTrain2, xTest)
    yTrain1 = yTrain1 / max(yTrain1)
    yTrain2 = yTrain1 / max(yTrain2)

    model = RandomForestRegressor(100)
    # model = KNeighborsRegressor(10)
    # model = SVR()
    model.fit(xTrain1, yTrain1)
    model.fit(xTrain2, yTrain2)
    scores = model.score(xTest, yTest)
    yPredicted = model.predict(xTest)
    yPredicted = sortPredict(yPredicted)

    print(scores)
    print(yPredicted)

    i = 0
    accurateElement = 0
    while i < len(yPredicted):
        if abs(round(yPredicted[i]) - yTest[i]) <= 3:
            accurateElement = accurateElement + 1
        i = i + 1
    accurateElement = accurateElement / len(yPredicted) * 100
    print("Accuracy: ")
    print(accurateElement)
    print("Difference between Predicted and Real: ")
    print(yPredicted - yTest)


def sortPredict(yPredicted):
    y = np.zeros(yPredicted.shape)

    maxValue = max(yPredicted)
    for index in range(len(yPredicted)):
        indexMin = yPredicted.argmin()
        y[indexMin] = index + 1
        yPredicted[indexMin] = maxValue + 1

    return y


# 归一化数据集
def normalizationTest(trainData, testData):
    trainDataNormalized = np.zeros(trainData.shape, dtype='float')
    testDataNormalized = np.zeros(testData.shape, dtype='float')

    for col in range(trainData.shape[1]):
        colTrainData = trainData[..., col]
        colTestData = testData[..., col]

        maxData = max(max(colTrainData), max(colTestData))
        minData = 0

        colTrainData = (colTrainData - minData) / (maxData - minData + 1)
        colTestData = (colTestData - minData) / (maxData - minData + 1)

        trainDataNormalized[..., col] = colTrainData
        testDataNormalized[..., col] = colTestData

    return trainDataNormalized, testDataNormalized

# 使用提供的测试集测试
def testModelTest():
    xTrain = np.array(xTrainPickle)
    yTrain = np.array(yTrainPickle)
    xTest = np.array(xTestPickle)
    yTest = np.array(yTestPickle)

    xTrain, xTest = normalizationTest(xTrain, xTest)
    yTrain = yTrain / max(yTrain)

    model = RandomForestRegressor(1000)
    # model = KNeighborsRegressor(10)
    # model = SVR()
    model.fit(xTrain, yTrain)
    scores = model.score(xTest, yTest)
    yPredicted = model.predict(xTest)
    yPredicted = sortPredict(yPredicted)

    print(scores)
    print(yPredicted)

    i = 0
    accurateElement = 0
    while i < len(yPredicted):
        if abs(round(yPredicted[i]) - yTest[i]) <= 3:
            accurateElement = accurateElement + 1
        i = i + 1
    accurateElement = accurateElement / len(yPredicted) * 100
    print("Accuracy: ")
    print(accurateElement)
    print("Difference between Predicted and Real: ")
    print(yPredicted - yTest)
