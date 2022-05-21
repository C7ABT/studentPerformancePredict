# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/3/17 19:09
# @Function: 将数据“摊平”，3个学期排在1行内

import pickle
import random
import matplotlib.pyplot as plt
import numpy
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd

def dataReshape():
    dataPreTreated = pickle.load(open('DataPreTreated.pkl', 'rb'))
    spendAmountArray = pickle.load(open('SpendKmeansData.pkl', 'rb'))
    scheduledAmountArray = pickle.load(open('ScheduledKmeansData.pkl', 'rb'))
    studyAmountArray = pickle.load(open('StudyKmeansData.pkl', 'rb'))
    lenDataPreTreated = len(dataPreTreated)
    # 探索聚类数量
    # 消费：4，学习：3，规律生活：3

    # MSE = []
    # for k in range(2, 20):
    #     km = KMeans(n_clusters=k)
    #     km.fit(spendAmountArray)
    #     MSE.append(km.inertia_)
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(2, 20), MSE, 'o-')
    # plt.xticks(range(0, 22, 1))
    # plt.grid(linestyle='--')
    # plt.xlabel("Number of Clusters Initialized")
    # plt.ylabel('SSE')
    # plt.title("SSE")
    # plt.show()
    # sns.despine()

    # 消费聚类分析
    kmSpend = KMeans(n_clusters=4)
    kmSpend.fit(spendAmountArray)
    print("消费聚类分析")
    print(pd.value_counts(kmSpend.labels_))
    # 规律生活聚类分析
    kmScheduled = KMeans(n_clusters=3)
    kmScheduled.fit(scheduledAmountArray)
    print("规律生活聚类分析")
    print(pd.value_counts(kmScheduled.labels_))
    # 学习聚类分析
    kmStudy = KMeans(n_clusters=3)
    kmStudy.fit(studyAmountArray)
    print("学习聚类分析")
    print(pd.value_counts(kmStudy.labels_))

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
