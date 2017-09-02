#!/usr/bin/python
# coding:utf-8
import profile

import numpy as np
import pandas as pd
from sklearn import svm


# label = digit.values[:, 0].astype(int)
# train = digit.values[:, 1:].astype(int)
# test_data = test.values[:, :].astype(int)


def getData():
    print '获取训练数据中。。。'
    trainData = pd.read_csv('train.csv')
    trainSet = trainData.values[:, 1:].astype(int)
    trainLabel = trainData.values[:, 0].astype(int)
    testData = pd.read_csv('test.csv')
    print '获取测试数据中'
    testSet = testData.values[:, :].astype(int)
    return trainSet, trainLabel, testSet


trainSet, trainLabel, testSet = getData()
# print 'train len: ', len(trainSet[0])
# print trainSet[:2]
# print 'trainLabel len: ', len(trainLabel)
# print trainLabel[:10]
# print 'testSet len: ', len(testSet)
# print testSet[:4]
svmClf = svm.SVC()
print 'SVM分类器训练中。。。'
svmClf.fit(trainSet, trainLabel)
print 'SVM测试算法中。。。'
re = svmClf.predict(testSet)
cnt = 1
tmpStr = ''
print '写入数据中'
with open('result.csv', 'w') as f:
    for element in re:
        tmpStr = str(cnt) + ',' + str(element) + '\n'
        cnt += 1
        print tmpStr
        f.write(tmpStr)
