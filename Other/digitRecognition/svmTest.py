#!/usr/bin/python
# coding:utf-8
import profile
import numpy as np
import pandas as pd
from sklearn import svm, model_selection, metrics


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
CNT = 200
X = trainSet[:CNT, :CNT]
y = trainLabel[:CNT]
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, random_state=10)
# raw_input()
svmClf = svm.SVC(gamma=0.001)
# print 'SVM分类器训练中。。。'
svmClf.fit(X_train, y_train)
# print 'SVM测试算法中。。。'
y_pre = svmClf.predict(X_test)
print '预测值: ', y_pre
print '实际值: ', y_test
print '正确率： ', metrics.accuracy_score(y_test, y_pre)
# cnt = 1
# tmpStr = ''
# print '写入数据中'
# with open('result.csv', 'w') as f:
#     for element in re:
#         tmpStr = str(cnt) + ',' + str(element) + '\n'
#         cnt += 1
#         print tmpStr
#         f.write(tmpStr)
