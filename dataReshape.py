# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/3/17 19:09
# @Function: 将数据“摊平”，3个学期排在1行内

import pickle

def dataReshape():
    dataPreTreated = pickle.load(open('dataPreTreated.pkl', 'rb'))
    dataToReshape = []

    for index in range(len(dataPreTreated)):
        dataToReshape.append(dataPreTreated[index])

    trainDataX1 = []
    trainDataY1 = []
    trainDataX2 = []
    trainDataY2 = []
    testDataX = []
    testDataY = []
    for index in range(int(len(dataToReshape) / 3)):
        tmp = []
        d1 = dataToReshape[index * 3]
        tmp += d1[3:] + [d1[2]]
        trainDataX1.append(tmp[:-1])
        trainDataY1.append(tmp[-1])

        tmp = []
        d2 = dataToReshape[index * 3 + 1]
        tmp += d2[3:] + [d2[2]]
        trainDataX2.append(tmp[:-1])
        trainDataY2.append(tmp[-1])

        tmp = []
        testData = dataToReshape[index * 3 + 2]
        tmp = tmp + testData[3:] + [testData[2]]
        testDataX.append(tmp[:-1])
        testDataY.append(tmp[-1])
    pickle.dump(trainDataX1, open('TrainX1.pkl', 'wb'))
    pickle.dump(trainDataY1, open('TrainY1.pkl', 'wb'))
    pickle.dump(trainDataX2, open('TrainX2.pkl', 'wb'))
    pickle.dump(trainDataY2, open('TrainY2.pkl', 'wb'))
    pickle.dump(testDataX, open('TestX.pkl', 'wb'))
    pickle.dump(testDataY, open('TestY.pkl', 'wb'))


# 使用提供的测试集测试
def dataReshapeTest():
    trainDataPreTreated = pickle.load(open('dataPreTreated.pkl', 'rb'))
    testDataPreTreated = pickle.load(open('TestDataPreTreated.pkl', 'rb'))
    trainDataToReshape = []
    testDataToReshape = []

    for index in range(len(trainDataPreTreated)):
        trainDataToReshape.append(trainDataPreTreated[index])
    # for index in range(len(testDataPreTreated)):
    #     testDataToReshape.append(testDataPreTreated[index])

    trainDataX = []
    trainDataY = []
    testDataX = []
    testDataY = []
    for index in range(int(len(trainDataToReshape) / 3)):
        tmp = []
        for i in range(3):
            d = trainDataToReshape[index * 3 + i]
            tmp += d[3:] + [d[2]]
        trainDataX.append(tmp[:-1])
        trainDataY.append(tmp[-1])

    for index in range(int(len(testDataToReshape) / 3)):
        tmp = []
        for i in range(3):
            d = testDataToReshape[index * 3 + i]
            tmp += d[3:] + [d[2]]
        testDataX.append(tmp[:-1])
        testDataY.append(tmp[-1])
    # pickle.dump(trainDataX, open('TrainX.pkl', 'wb'))
    # pickle.dump(trainDataY, open('TrainY.pkl', 'wb'))
    # pickle.dump(testDataX, open('TestX.pkl', 'wb'))
    # pickle.dump(testDataY, open('TestY.pkl', 'wb'))
    index = int(len(trainDataX) / 5 * 4)
    pickle.dump(trainDataX[0: index], open('TrainX.pkl', 'wb'))
    pickle.dump(trainDataY[0: index], open('TrainY.pkl', 'wb'))
    pickle.dump(trainDataX[index:], open('TestX.pkl', 'wb'))
    pickle.dump(trainDataY[index:], open('TestY.pkl', 'wb'))
