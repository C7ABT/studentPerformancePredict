# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/1/10 16:32
# @Function: 数据仅导出，不计算方差等值

import numpy as np
import pickle
from sklearn.cluster import KMeans
from matplotlib.pyplot import plot as plt
def preTreat():
    fileRank = open('成绩.txt', 'r')
    fileLibrary = open('图书馆门禁.txt', 'r')
    fileBookBorrow = open('借书.txt', 'r')
    fileSpend = open('消费.txt', 'r')

    BOOKTYPEAMOUNT = 43

    studentAmount = 538
    spendAmountArray = np.zeros((studentAmount, 3), dtype=float)
    scheduledAmountArray = np.zeros((studentAmount, 2), dtype=int)
    studyAmountArray = np.zeros((studentAmount, 3), dtype=float)

    data = []

    '''
    读入成绩
    学期 学号 排名
    '''
    # Repeat this to ignore the title
    fileRank.readline()
    line = fileRank.readline()
    while line:
        temp = []
        pre = 0
        lenLine = len(line)
        # For each character in each line, covert chr -> number and save it in Array.
        # '\n' ignored
        # [pre:end) -> temp
        while pre < lenLine - 2:
            if line[pre] != '\t':
                end = pre + 1
                while end < lenLine - 1:
                    if line[end] == '\t':
                        temp += [int(line[pre:end])]  # We got the content of a particular field
                        pre = end + 1
                        break
                    else:
                        end = end + 1
            else:
                pre = pre + 1
                # end = pre + 1
        data += [temp]
        line = fileRank.readline()
    '''
    学期  学号  排名
    '''
    # print(data)
    # Sort by 学号 first, 学期 second
    data = sorted(data, key=lambda x: (x[1], x[0]))

    '''
    读入图书馆门禁数(学期计)
    '''

    '''
    0    1     2                  
    学期  学号  排名  
    3
    入馆总数
    '''

    # Repeat this to ignore the title
    line = fileLibrary.readline()
    line = fileLibrary.readline()
    zero1 = np.zeros(1, int).tolist()
    i = 0
    while i < len(data):
        data[i] += zero1  # Expand the dimension
        i = i + 1

    while line:
        (semester, studentID, date, time, endTime) = line.split('\t')
        semester = int(semester)
        studentID = int(studentID)
        # studentID as the main Key, semester as the secondary key in data
        index = (studentID - 1) * 3 + semester - 1
        # total time in a semester
        data[index][3] += 1
        # 学习行为聚类数据
        kMeansArrayIndex = int(studentID) - 1
        studyAmountArray[kMeansArrayIndex][0] += 1  # 图书馆访问次数

        line = fileLibrary.readline()

    '''
    读入借书
    '''

    BOOKCASES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'TB', 'TD', 'TE', 'TF', 'TG', 'TH', 'TJ', 'TK', 'TL', 'TM', 'TN', 'TP', 'TQ', 'TS', 'TT', 'TU', 'TV',
                 'U', 'V', 'X', 'Y', 'Z', 'OO']

    '''
    0    1     2        
    学期  学号  排名
    3           
    入馆总数    
    4            
    图书借阅总次数
    '''
    bookInfo = pickle.load(open('BookInfo.pkl', 'rb'))
    zeroBookBorrowedAmount = np.zeros(1, int).tolist()
    i = 0
    bookBorrowedAmountOffset = 4
    while i < len(data):
        data[i] += zeroBookBorrowedAmount
        i = i + 1

    line = fileBookBorrow.readline()
    line = fileBookBorrow.readline()

    while line:
        (semester, studentID, bookName, date, endTime) = line.split('\t')
        index = (int(studentID) - 1) * 3 + int(semester) - 1
        # # book not in BOOKCASES
        # if bookName not in bookInfo.keys():
        #     data[index][42 + bookBorrowedAmountOffset] += 1
        # else:
        #     pos = BOOKCASES.index(bookInfo[bookName])
        #     data[index][pos + bookBorrowedAmountOffset] += 1
        data[index][bookBorrowedAmountOffset] += 1
        # 学习行为聚类数据
        kMeansArrayIndex = int(studentID) - 1
        studyAmountArray[kMeansArrayIndex][1] += 1  # 图书借阅量

        line = fileBookBorrow.readline()

    '''
    0    1     2        
    学期  学号  排名
    3           
    入馆总数    
    4             
    图书借阅总次数
    5    6   7   8    9   10  11                
    超市 打印 交通 教室 食堂 宿舍 图书馆--消费总额    
    '''

    SPENDCASES = ['超市', '打印', '交通', '教室', '食堂', '宿舍', '图书馆']
    SPENDTYPEAMOUNT = 7
    zeroSpendTypeAmount = np.zeros(SPENDTYPEAMOUNT, int).tolist()
    i = 0
    # data[5:11] -> BOOKCASES[0:6]
    spendTypeOffset = 5
    while i < len(data):
        data[i] += zeroSpendTypeAmount
        i = i + 1

    line = fileSpend.readline()
    line = fileSpend.readline()
    money = 0.0
    while line:
        (semester, studentID, spendName, date, endTime, moneySpent) = line.split('\t')
        money = max(money, float(moneySpent))
        # print(semester, studentID, spendName, date, endTime, moneySpent)
        index = (int(studentID) - 1) * 3 + int(semester) - 1
        pos = SPENDCASES.index(spendName)
        # 索引是否正确？？
        data[index][pos + spendTypeOffset] += float(moneySpent)

        kMeansArrayIndex = int(studentID) - 1
        # 消费聚类数据
        spendAmountArray[kMeansArrayIndex][0] += float(moneySpent)  # 消费总额
        spendAmountArray[kMeansArrayIndex][1] = max(float(moneySpent), spendAmountArray[kMeansArrayIndex][1])  # 单次消费最大值
        spendAmountArray[kMeansArrayIndex][2] += 1  # 消费次数
        # 规律生活聚类数据
        if spendName == '食堂':
            mealHour = int(endTime[0:2])
            mealMinute = int(endTime[2:4])
            mealSecond = int(endTime[4:])
            # 6-8, 11:30-13:00, 17:30-19:00
            if (mealHour == 6) or (mealHour == 7) or (mealHour == 11 and mealMinute >= 30) or (mealHour == 12) or (
                    mealHour == 17 and mealMinute >= 30) or (mealHour == 18):
                scheduledAmountArray[kMeansArrayIndex][0] += 1  # 按时吃饭
            if (mealHour == 6) or (mealHour == 7):
                scheduledAmountArray[kMeansArrayIndex][1] += 1  # 早起
        # 学习行为聚类数据
        kMeansArrayIndex = int(studentID) - 1
        if spendName == '教室' or spendName == '图书馆' or spendName == '打印':
            studyAmountArray[kMeansArrayIndex][2] += float(moneySpent)  # 教室、图书馆、打印学习资料消费金额

        line = fileSpend.readline()

    # print(np.array(data).shape)
    # 1614行，3个学期，一共538人
    pickle.dump(data, open('DataPreTreated.pkl', 'wb'))
    pickle.dump(spendAmountArray, open('SpendKmeansData.pkl', 'wb'))
    pickle.dump(scheduledAmountArray, open('ScheduledKmeansData.pkl', 'wb'))
    pickle.dump(studyAmountArray, open('StudyKmeansData.pkl', 'wb'))
