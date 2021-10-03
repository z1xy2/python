from typing import List
import numpy as np
from functools import reduce
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec
#创建词汇表
def createVocabList(postingList):
    vocabSet=set([])
    for list1 in postingList:
        vocabSet=vocabSet|set(list1)
    return list(vocabSet)
#词集模型
#把一行词转换为向量方便操作
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print(f"{word}is not in vocabulary")
    return returnVec
def trainNB0(trainMatrix,trainCategory):
    numDocs = len(trainMatrix)                            #计算训练矩阵行数 文档数
    numWords = len(trainMatrix[0])                            #计算训练矩阵列数 每篇文档的词条数
    pAbusive = sum(trainCategory)/numDocs                     #文档属于侮辱类的概率=侮辱侮辱文档总数/文档总数
    #计算每个单词在侮辱类和费侮辱类出现的概率
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)    #创建numpy.zeros数组,词条出现数初始化为0 每个单词出现的频次
    p0Denom = 0; p1Denom = 0                            #分母初始化为0 总单词数
    for i in range(numDocs):
        if trainCategory[i] == 1:                           #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                                                #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    print(p0Num)
    p1Vect = p1Num/p1Denom                                      
    p0Vect = p0Num/p0Denom         
    return p0Vect,p1Vect,pAbusive                            #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

def classifyNB(testVec, p0V, p1V, pAb):
    p1,p0=pAb,1-pAb
    for index in range(len(testVec)) :
        if  testVec[index]==1:
            p1=p1V[index]*p1
            p0 = p0V[index] * p0
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0 : return 1 
    else : return 0

if __name__=='__main__':
    postingList,classVec=loadDataSet()
    myVocabList=createVocabList(postingList)
    print(myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    #先把输入的单词向量化
    testEntry = ['love', 'my', 'dalmation']		
    testVec=setOfWords2Vec(myVocabList,testEntry)
    a=classifyNB(np.array(testVec),np.array(p0V), np.array(p1V), pAb)#传入数据进行计算 0非侮辱 1侮辱
    if(a):
        print(f'{testEntry}包含侮辱词汇')
    else:
        print(f'{testEntry}不包含侮辱词汇')