# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/1/10 16:32
# @Function: 数据仅导出，不计算方差等值

import numpy as np
import pickle


def preTreat():
    fileRank = open('成绩.txt', 'r')
    fileLibrary = open('图书馆门禁.txt', 'r')
    fileBookBorrow = open('借书.txt', 'r')
    fileSpend = open('消费.txt', 'r')

    BOOKTYPEAMOUNT = 43

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
    0    1     2    3              
    学期  学号  排名  入馆总数
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
    4-46             
    每类书借阅总次数
    '''
    bookInfo = pickle.load(open('BookInfo.pkl', 'rb'))
    zeroBookTypeAmount = np.zeros(BOOKTYPEAMOUNT, int).tolist()
    i = 0
    # data[4:46] -> BOOKCASES[0:42]
    bookTypeOffset = 4
    while i < len(data):
        data[i] += zeroBookTypeAmount
        i = i + 1

    line = fileBookBorrow.readline()
    line = fileBookBorrow.readline()

    while line:
        (semester, studentID, bookName, date, endTime) = line.split('\t')
        index = (int(studentID) - 1) * 3 + int(semester) - 1
        # book not in BOOKCASES
        if bookName not in bookInfo.keys():
            data[index][42 + bookTypeOffset] += 1
        else:
            pos = BOOKCASES.index(bookInfo[bookName])
            data[index][pos + bookTypeOffset] += 1
        line = fileBookBorrow.readline()

    '''
    0    1     2        
    学期  学号  排名
    3           
    入馆总数    
    4-46             
    每类书借阅总次数
    47   48  49  50   51  52  53                
    超市 打印 交通 教室 食堂 宿舍 图书馆--消费总额    
    '''

    SPENDCASES = ['超市', '打印', '交通', '教室', '食堂', '宿舍', '图书馆']
    SPENDTYPEAMOUNT = 7
    zeroSpendTypeAmount = np.zeros(SPENDTYPEAMOUNT, int).tolist()
    i = 0
    # data[47:53] -> BOOKCASES[0:6]
    spendTypeOffset = 47
    while i < len(data):
        data[i] += zeroSpendTypeAmount
        i = i + 1

    line = fileSpend.readline()
    line = fileSpend.readline()

    while line:
        (semester, studentID, spendName, date, endTime, moneySpent) = line.split('\t')
        # print(semester, studentID, spendName, date, endTime, moneySpent)
        index = (int(studentID) - 1) * 3 + int(semester) - 1
        pos = SPENDCASES.index(spendName)
        # 索引是否正确？？
        data[index][pos + spendTypeOffset] += float(moneySpent)
        line = fileSpend.readline()

    # print(np.array(data).shape)
    # 1614行，3个学期，一共538人
    pickle.dump(data, open('DataPreTreated.pkl', 'wb'))

