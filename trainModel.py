# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/3/15 17:10
# @Function: 模型训练

import pickle

from keras.optimizers import Adam
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

    xTrainLstm = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], 1))
    xTestLstm = xTest.reshape((xTest.shape[0], xTest.shape[1], 1))
    model = Sequential()

    model.add(LSTM(units=5, dropout=0.1, activation='relu'))
    # model.add(LSTM(units=50, dropout=0.1, return_sequences=True, activation='relu'))
    # model.add(LSTM(units=50, dropout=0.1, activation='relu'))

    model.add(Dense(units=1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(xTrainLstm, yTrain, batch_size=32, epochs=20)
    yPredicted = model.predict(xTestLstm)

    # yPredicted = (yPredicted > 0.5).astype(int)
    yPredicted = np.reshape(yPredicted, [-1])
    yTest = np.reshape(yTest, [-1])
    print(yPredicted)
    print(yTest)

    print(accuracy_score(yPredicted, yTest))
    print(classification_report(yPredicted, yTest))
