import operator
import numpy as np
'''过滤网站恶意留言，侮辱性 1，非侮辱性 0 ，创建以下实验样本'''
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec
#创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])#创建一个空表
    for document in dataSet:
        vocabSet = vocabSet |set(document)#创建两个集合的并集
    return list(vocabSet)
#将文档词条转化为词向量
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList) #创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            #index函数。在字符串里找到字符第一次出现的位置
            returnVec[vocabList.index(word)] +=1 #文档的词袋模型，每个单词可以出现多次
        else:print('the word %s is not in my Vocabulary'%word)
    return returnVec

#朴素贝叶斯分类器训练函数，从词向量计算概率
def trainNBO(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)#用来统计两类数据中，各词的频次
    p0Denom = 2.0 #用于统计0类中的总数
    p1Denom = 2.0 #用于统计1类中的总数
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom) #在1类中，每个词发生的概率
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#朴素贝叶斯分类器
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts ,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0v,p1v,pAb = trainNBO(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as',classifyNB(thisDoc,p0v,p1v,pAb))

    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as',classifyNB(thisDoc,p0v,p1v,pAb))
testingNB()
















