# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/3/17 19:09
# @Function: 将数据“摊平”，3个学期排在1行内

import pickle

import numpy


def dataReshape():
    trainDataPreTreated = pickle.load(open('TrainDataPreTreated.pkl', 'rb'))
    testDataPreTreated = pickle.load(open('TestDataPreTreated.pkl', 'rb'))

    trainData = []
    testData = []

    for index in range(len(trainDataPreTreated)):
        trainData.append(trainDataPreTreated[index])
    for index in range(len(testDataPreTreated)):
        testData.append(testDataPreTreated[index])
    '''
    把三个学期的成绩合在一起，不要学期，学号，排名放在最后
    '''
    trainDataX = []
    trainDataY = []
    for index in range(int(len(trainData) / 3)):
        tmp = []
        for i in range(3):
            d = trainData[index * 3 + i]
            tmp = tmp + d[3:] + [d[2]]
        trainDataX.append(tmp[:-1])
        trainDataY.append(tmp[-1])
    pickle.dump(trainDataX, open('TrainX.pkl', 'wb'))
    pickle.dump(trainDataY, open('TrainY.pkl', 'wb'))

    testDataX = []
    testDataY = []
    for index in range(int(len(testData) / 3)):
        tmp = []
        for i in range(3):
            d = testData[index * 3 + i]
            tmp = tmp + d[3:] + [d[2]]
        testDataX.append(tmp[:-1])
        testDataY.append(tmp[-1])
    pickle.dump(testDataX, open('TestX.pkl', 'wb'))
    pickle.dump(testDataY, open('TestY.pkl', 'wb'))
