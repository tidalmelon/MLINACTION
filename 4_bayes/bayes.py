#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Date: 2017/06/29 16:26:31
"""
from numpy import *


def loadDataSet():
    """
    返回文档集合
    """
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
    ]

    # 0 代表正常文字 1 代表侮辱性文字
    classVec = [0, 1, 0, 1, 0, 1, 0, 1]

    return postingList, classVec

def createVocabList(dataset):
    """
    包含在所有文档中出现的不重复词的列表
    """
    vocabSet = set([])
    for document in dataset:
        # 集合 并
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    vacabList: 词汇标
    inputSet: 一个文档
    return: 文档向量(与词汇表等长): 1 存在 0 不存在
    注解：朴素bayes通常有两种实现方式，bernoulli模型(
    不考虑词在文档中出现的次数，只考虑出不出现，相当于假设词是等权重的啦)，
    多项式模型(考虑词在文档中出现的次数)
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print 'the word: %s not in my vocabList' % word
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    trainMatrix: 文档矩阵
    trainCategory: 对应每篇文档的类别标签
    """

    # 训练文档数量
    numTrainDocs = len(trainMatrix)
    # 词数量, 每篇文章都编程了长度一致的向量，如setOfWords2Vec 所示。
    numWords = len(trainMatrix[0])
    # 侮辱性: 先验概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    # 初始化概率
    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numTrainDocs):
        # 侮辱性
        if trainCategory[i] == 1:
            # 向量相加

            # 分类1与某个词 同时出现
            p1Num += trainMatrix[i]
            # 分类1与所有词 同时出现
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # change to log()
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive



if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print myVocabList
