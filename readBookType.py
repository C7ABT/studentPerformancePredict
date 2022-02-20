# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/1/10 16:05
# @Function: Read and save book type

import pickle

def readBookType(isDebug = False):
    bookInfo = dict()
    errInfo = dict()
    bookTypeList = []

    with open('图书类别.txt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.replace('\n', '')
            (bookNumber, bookType) = line.split('\t')
            # bookNumber must be a number
            if not bookNumber.isdigit():
                errInfo[bookNumber] = bookType
                continue

            bookInfo[bookNumber] = bookType
            if bookType not in bookTypeList:
                bookTypeList.append(bookType)

    bookTypeList.sort()

    if isDebug:
        for type in bookTypeList:
            print("bookType: " + type, end=' ')
        print(errInfo)


    pickle.dump(bookInfo, open('BookInfo.pkl', 'wb'))
    pickle.dump(bookTypeList, open('BookClass.pkl', 'wb'))

    bookInfo = pickle.load(open('BookInfo.pkl', 'rb'))
    bookTypeList = pickle.load(open('BookClass.pkl', 'rb'))
    return bookInfo, bookTypeList






