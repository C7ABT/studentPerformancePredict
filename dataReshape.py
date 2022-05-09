# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/3/17 19:09
# @Function: 将数据“摊平”，3个学期排在1行内

import pickle
import random


def dataReshape():
    dataPreTreated = pickle.load(open('DataPreTreated.pkl', 'rb'))
    lenDataPreTreated = len(dataPreTreated)

    data = []
    for index in range(lenDataPreTreated):
        data.append(dataPreTreated[index])

    '''
    把三个学期的成绩合在一起，不要学期，学号，成绩放在最后
    '''

    xData = []
    yData = []
    for index in range(int(len(data) / 3)):
        tmp = []
        for i in range(3):
            d = data[index * 3 + i]
            tmp = tmp + d[3:] + [d[2]]
        xData.append(tmp[:-1])
        yData.append(tmp[-1])

    # 二分类，80%之后的为不通过
    passLine = int(len(yData) / 5 * 4)
    for i in range(len(yData)):
        if yData[i] <= passLine:
            yData[i] = 0
        else:
            yData[i] = 1
    index = int(len(xData) / 5 * 4)
    # 随机抽取20%数据作为测试集，其余数据作为训练集
    testIndex = random.sample(range(len(xData)), len(xData) - index)
    xDataTest = []
    yDataTest = []
    for i in testIndex:
        xDataTest.append(xData[i])
        yDataTest.append(yData[i])
    xData = [xData[i] for i in range(0, len(xData), 1) if i not in testIndex]
    yData = [yData[i] for i in range(0, len(yData), 1) if i not in testIndex]

    pickle.dump(xData, open('TrainX.pkl', 'wb'))
    pickle.dump(yData, open('TrainY.pkl', 'wb'))
    pickle.dump(xDataTest, open('TestX.pkl', 'wb'))
    pickle.dump(yDataTest, open('TestY.pkl', 'wb'))

    #
    # pickle.dump(xData[:index], open('TrainX.pkl', 'wb'))
    # pickle.dump(yData[:index], open('TrainY.pkl', 'wb'))
    # pickle.dump(xData[index:], open('TestX.pkl', 'wb'))
    # pickle.dump(yData[index:], open('TestY.pkl', 'wb'))
