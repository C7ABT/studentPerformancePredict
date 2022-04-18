from checkBookType import checkBookType
from readBookType import readBookType
from dataPreTreat import preTreat
from dataReshape import dataReshape, dataReshapeTest
from trainModel import testModel, testModelTest


def main():
    # readBookType()
    preTreat(isTrain=True, isDebug=True)
    preTreat(isTrain=False, isDebug=True)
    dataReshape()
    # dataReshapeTest()
    testModel()

if __name__ == '__main__':
    main()
