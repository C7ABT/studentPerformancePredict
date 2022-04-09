from checkBookType import checkBookType
from readBookType import readBookType
from dataPreTreat import preTreat
from dataReshape import dataReshape
from trainModel import testModel


def main():
    # readBookType()
    preTreat(isTrain=True, isDebug=True)
    preTreat(isTrain=False, isDebug=True)
    dataReshape()
    testModel()


if __name__ == '__main__':
    main()
