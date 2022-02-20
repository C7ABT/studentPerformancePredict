# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/1/10 16:32
# @Function: Data pre treat

import numpy as np
import pickle


def preTreat():
    fileRank = open('成绩.txt', 'r')
    fileLibrary = open('图书馆门禁.txt', 'r')
    fileBookBorrow = open('借书.txt', 'r')
    fileSpend = open('消费.txt', 'r')

    BOOKTYPEAMOUNT = 43

    '''
    学期 学号 图书馆门禁次数 食堂总消费 交通总消费 宿舍总消费 超市总消费 书类别  排名
    '''
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
                        temp += int(line[pre:end])  # We got the content of a particular field
                        pre = end + 1
                        break
                    else:
                        end = end + 1
            else:
                pre = pre + 1
                # end = pre + 1
        data += temp
        line = fileRank.readline()
    '''
    学期  学号  排名
    '''
    # Sort by 学号 first, 学期 second
    data = sorted(data, key=lambda x: (x[1], x[0]))

    '''
    读入图书馆门禁数(学期计)
    '''

    '''
    0    1     2    3          4-7                8-11               12-28       
    学期  学号  排名  入馆总数    月平均/方差/max/min  日平均/方差/max/min  06-22入馆数
    '''

    # Repeat this to ignore the title
    line = fileLibrary.readline()
    line = fileLibrary.readline()
    zero26 = np.zeros(26, int).tolist()
    i = 0
    while i < len(data):
        data[i] += zero26   # Expand the dimension
        i = i + 1

    # How much time/(student * day), 31 days each month
    # 6 months, 31 days
    data3Semesters = np.zeros(shape=(len(data), 6, 31)).tolist()
    while line:
        (semester, studentID, date, time, endTime) = line.split('\t')
        semester = int(semester)
        studentID = int(studentID)
        # studentID as the main Key, semester as the secondary key in data
        index = (studentID - 1) * 3 + semester - 1
        # total time in a semester
        data[index][3] += 1
        # data[12:28] -> 06-22
        timeOffset = 6

        hour = int(time[0:2])
        data[index][hour + timeOffset] += 1
        month = int(date[0:2])

        day = int(date[2:])
        '''
        1/3 semester data3Semesters[index][0-4] -> September-January, 9~0 10~1 11~2 12~3 1~4
        2   semester data3Semesters[index][0-5] -> February-July
        '''
        if semester != 2:
            data3Semesters[index][(month - 9) % 12][day - 1] += 1
        else:
            data3Semesters[index][month - 2][day - 1] += 1
        line = fileLibrary.readline()

    '''
    Calculate average, variance, max, min
    '''
    n_1 = 5  # 1/3 semester, 5 months
    n_2 = 6  # 2   semester, 6 months
    i = 0
    while i < len(data):
        if i % 3 == 1:  # 是否为第二学期
            n = n_2
        else:
            n = n_1
        monthAverage = data[i][3] / n
        """
        TBD: 日平均值偏小？？有些日期对应取值为0
        """
        dayAverage = data[i][3] / (n * 31)
        # amount for months
        monthAmount = np.zeros(n, int).tolist()
        # amount for days
        dayAmount = np.zeros(n * 31, int).tolist()
        indexMonth = 0
        indexDaySemester = 0
        while indexMonth < n:
            monthAmount[indexMonth] = int(sum(data3Semesters[i][indexMonth]))
            indexDayMonth = 0
            while indexDayMonth < 31:
                dayAmount[indexDaySemester] = int(data3Semesters[i][indexMonth][indexDayMonth])
                indexDaySemester = indexDaySemester + 1
                indexDayMonth = indexDayMonth + 1
            indexMonth = indexMonth + 1
        monthMax = max(monthAmount)
        dayMax = max(dayAmount)
        monthMin = min(monthAmount)
        dayMin = min(dayAmount)
        monthVariance = np.var(monthAmount)
        dayVariance = np.var(dayAmount)
        data[i][4:12] = monthAverage, monthVariance, monthMax, monthMin, dayAverage, dayVariance, dayMax, dayMin
        i = i + 1
    '''
    读入借书
    '''

    BookClass = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'TB', 'TD', 'TE', 'TF', 'TG', 'TH', 'TJ', 'TK', 'TL', 'TM', 'TN', 'TP', 'TQ', 'TS', 'TT', 'TU', 'TV',
                 'U', 'V', 'X', 'Y', 'Z', 'OO']

    '''
    0    1     2    3-28             29-71             72-75              76-79    
    学期  学号  排名  fileLibrary属性   每类书借阅总次数    月平均/方差/max/min  日平均/方差/max/min
    '''
    # How much book each person borrows each day
    data2 = np.zeros(shape=(len(data), 6, 31)).tolist()
    bookInfo = pickle.load(open('BookInfo.pkl', 'rb'))
    zeroBookTypeAmount = np.zeros(BOOKTYPEAMOUNT, int).tolist()
    i = 0
    timeOffset = 30  # 书类别偏移
    while i < len(data):
        data[i] += zeroBookTypeAmount
        i = i + 1

    line = fileBookBorrow.readline()
    line = fileBookBorrow.readline()

    while line:
        (semester, studentID, bookName, date, endTime) = line.split('\t')
        index = (int(studentID) - 1) * 3 + int(semester) - 1
        month = int(date[:2])
        day = int(date[2:])
        if int(semester) != 2:
            data2[index][(month - 9) % 12][day - 1] += 1
        else:
            data2[index][month - 2][day - 1] += 1
        if bookName not in bookInfo.keys():
            data[index][42 + timeOffset - 1] += 1
        else:
            i = 0
            while i < len(BookClass) - 1:
                if BookClass[i] == bookInfo[bookName]:
                    break
                i = i + 1
            data[index][i + timeOffset - 1] += 1
        line = fileBookBorrow.readline()
    i = 0
    '''
    计算月 日 放进 data
    '''
    zeros8 = np.zeros(8, int).tolist();
    while i < len(data):
        data[i] += zeros8
        i = i + 1
    i = 0
    while i < len(data):
        if i % 3 == 1:
            n = n_2
        else:
            n = n_1
        num = sum(data[i][timeOffset - 1:])  # 某个人某学期总借书量
        monthAverage = num / n;
        dayAverage = num / (n * 31);
        monthAmount = np.zeros(n, int).tolist();
        dayAmount = np.zeros(n * 31, int).tolist();
        indexMonth = 0  # 第j个月
        indexDaySemester = 0  # 该学期第l天
        while indexMonth < n:
            monthAmount[indexMonth] = int(sum(data2[i][indexMonth]))
            indexDayMonth = 0  # 第j月中的k天
            while indexDayMonth < 31:
                dayAmount[indexDaySemester] = int(data2[i][indexMonth][indexDayMonth])
                indexDaySemester = indexDaySemester + 1
                indexDayMonth = indexDayMonth + 1
            indexMonth = indexMonth + 1
        monthMax = max(monthAmount);
        dayMax = max(dayAmount);
        monthMin = min(monthAmount);
        dayMin = min(dayAmount);
        monthVariance = np.var(monthAmount);
        dayVariance = np.var(dayAmount);
        data[i][72:] = monthAverage, monthVariance, monthMax, monthMin, dayAverage, dayVariance, dayMax, dayMin;
        i = i + 1

    # 整理data
    train_predata_x = []
    train_y = []
    for i in range(538):
        tmp = [data[i * 3][2], data[i * 3 + 1][2]] + [data[i * 3][1]] + data[i * 3][3:] + data[i * 3 + 1][3:] + \
              data[i * 3 + 2][3:]

        train_predata_x.append(tmp)
        train_y.append(data[i * 3 + 2][2])
    # =============================================================================
    #     if item[0] == 3:
    #         train_y.append(item[2])
    #     if item[1] == pre_stu_id:
    #         stu_info.extend(item[2:])
    #     else:
    #         data_zhengli.append(stu_info)
    #         stu_info = item[1:]
    #         pre_stu_id = item[1]
    # =============================================================================

    ''' 学期、学号、排名、门禁、书籍信息 '''
    # pickle.dump(data, open('data_pre.pkl', 'wb'))
    # pickle.dump(train_predata_x, open('train_predata_x.pkl', 'wb'))

    return train_predata_x, train_y



