#~/usr/bin/python
# coding:utf-8
import random
import os
import re
import numpy
import math
import profile
import threading
import time


def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def createVocabList(docList):
    vocabSet = set([])
    for document in docList:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def getFullTestVec():
    print 'starting get full test Vec ......'
    docList = []
    classList = []
    basepath = os.getcwd()
    hampath = basepath + '/ham/'
    filesNameList = os.listdir(hampath)
    for eachFile in filesNameList:
        with open(hampath + eachFile, 'r') as f:
            docList.append(textParse(f.read()))
            classList.append(0)
    spampath = basepath + '/spam/'
    filesNameList = os.listdir(spampath)
    for eachFile in filesNameList:
        with open(spampath + eachFile, 'r') as f:
            docList.append(textParse(f.read()))
            classList.append(1)
    vocabList = createVocabList(docList)
    print 'over geting full text!!!'
    return docList, vocabList, classList


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):  # 训练参数,得到一个参数矩阵，对应着各个单词对应分类的出现频率
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = numpy.ones(numWords)
    p1Num = numpy.ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = numpy.log(p1Num / p1Denom)
    p0Vec = numpy.log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + numpy.log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + numpy.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def holdOutCrossValidation(docList, vocabList, classList):
    testList = []
    testClass = []
    trainList = docList[:]
    trainClass = classList[:]
    lenOfTestList = len(docList) / 5
    lenOfDocList = len(docList)
    print 'start geting train words vec and test words vec......'
    for index in range(lenOfTestList):
        randomIndex = int(random.uniform(0, lenOfDocList))
        lenOfDocList -= 1
        testList.append(docList[randomIndex])
        testClass.append(classList[randomIndex])
        del(trainList[randomIndex])
        del(trainClass[randomIndex])
    print 'start calc args......'
    tmpCnt = 0
    sumCnt = len(docList)
    trainMat = []
    for eachDoc in trainList:
        trainMat.append(setOfWords2Vec(vocabList, eachDoc))
        tmpCnt += 1
        print tmpCnt, ' / ', sumCnt
    p0Vec, p1Vec, pSpam = trainNB0(
        numpy.array(trainMat), numpy.array(trainClass))
    print 'p0: ', p0Vec
    errorCnt = 0
    print 'start calc cross validation......'
    for indexOfTestList in range(0, len(testList)):
        eachDocMat = setOfWords2Vec(vocabList, testList[indexOfTestList])
        if classifyNB(numpy.array(eachDocMat), p0Vec, p1Vec, pSpam) != testClass[indexOfTestList]:
            errorCnt += 1
    print 'len: ', len(trainList)
    return float(errorCnt) / len(testList)


class Test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # self._run_num = num

    def run(self):
        global mutex, docList_G, vocabList_G, classList_G
        threadname = threading.currentThread().getName()

        # for x in xrange(0, int(self._run_num)):
        print 'thread name: ', threadname
        mutex.acquire()
        holdOutCrossValidation(docList_G, vocabList_G, classList_G)
        mutex.release()


global docList_G, vocabList_G, classList_G, mutex
docList_G, vocabList_G, classList_G = getFullTestVec()

threads = []
num = 8
mutex = threading.Lock()

for x in xrange(0, num):
    threads.append(Test())

for t in threads:
    t.start()

for t in threads:
    t.join()


# holdOutCrossValidation(docList, vocabList, classList)
