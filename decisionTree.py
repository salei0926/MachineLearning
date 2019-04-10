from math import log
import operator

def calcShannonEnt(dataSet):  #计算数据的经验熵H(D)
    numEntries = len(dataSet)  #数据条数
    labelCounts={}
    for featVec in dataSet:
        currentLabel = featVec[-1]  #每行最后一个字符表类别
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1   #统计有多少个类，以及每个类的数量
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries   #计算单个类的熵值
        shannonEnt -=prob*log(prob,2)    #累加每个类的熵值
    return shannonEnt
def createDataSet1():  #创造示例数据
    dataSet = [['长','粗','男'],
               ['短','粗','男'],
               ['短', '粗', '男'],
               ['长','细','女'],
               [ '短','细','女'],
               ['长','粗','女'],
               ['短','粗','女'],
               ['长','粗','女'],]
    labels = ['头发','声音']
    return dataSet,labels

def splitDataSet(dataSet,axis,value):  #按某个特征分类后的数据
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):    #选择最优的分类特征
    numFeatures = len(dataSet[0]) - 1 #特征数
    baseEntropy = calcShannonEnt(dataSet)   #原始熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)  #按特征分类后的熵
        infoGain = baseEntropy -newEntropy     #信息增益
        if(infoGain>bestInfoGain):             #若按某特征值划分后，熵值减少最大，则此特征为最优分类特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):   #按分类后的类别数量排序，如最后分类为2男1女，则判定为男
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 若所有实例属于同一类c,则T为单节点树，并将类c作为该节点的类标记
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #若特征集为空，则T为单节点树，并将实例中实例数最大的类c作为该节点的类标记
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  #选择最优特征
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

if __name__=='__main__':
    dataSet,labels=createDataSet1()
    print(createTree(dataSet,labels))

















