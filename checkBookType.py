# -*- coding: utf-8 -*-
# @Author  : Connor Zhao
# @Time    : 2022/1/10 15:39
# @Function: Check if "图书类别.txt" has error

def checkBookType(isDebug = False):
    book = dict()
    classType = []
    duplicateNumber = []

    with open('图书类别.txt', encoding='utf-8') as f:
        f.readline()
        for line in f:
            (bookNumber, bookType) = line.split('\t')
            if bookNumber not in book.keys():
                book[bookNumber] = [bookType]
            else:
                book[bookNumber].append(bookType)
                if isDebug:
                    print(bookNumber)
                duplicateNumber.append(bookNumber)
            if bookType not in classType:
                classType.append(bookType)

    # See if we have any book with various types
    duplicateTypeList = []
    for index in duplicateNumber:
        temp_list = book[index]
        for dupicateType in temp_list[1:]:
            if dupicateType != temp_list[0]:
                duplicateTypeList.append(book[index])
                print(index + " has various types: " + book[index])
                break
    # Result: Same ID means same Type


