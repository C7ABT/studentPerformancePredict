# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/3/17 19:09
# @Function: 将数据“摊平”，3个学期排在1行内

import pickle

def dataReshape():
    dataPreTreated = pickle.load(open('DataPreTreated.pkl', 'rb'))
    lenDataPreTreated = len(dataPreTreated)

    data = []
    for index in range(lenDataPreTreated):
        data.append(dataPreTreated[index])

    '''
    把三个学期的成绩合在一起，不要学期，学号，成绩放在最后
    '''

    xTrain = []
    yTrain = []
    for index in range(int(len(data) / 3)):
        tmp = []
        for i in range(3):
            d = data[index * 3 + i]
            tmp = tmp + d[3:] + [d[2]]
        xTrain.append(tmp[:-1])
        yTrain.append(tmp[-1])

    index = int(len(xTrain) / 5 * 4)
    pickle.dump(xTrain[:index], open('TrainX.pkl', 'wb'))
    pickle.dump(yTrain[:index], open('TrainY.pkl', 'wb'))
    pickle.dump(xTrain[:index], open('TestX.pkl', 'wb'))
    pickle.dump(yTrain[:index], open('TestY.pkl', 'wb'))

