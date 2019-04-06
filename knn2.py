import operator
import numpy as np
from os import listdir
def classify0(inX,dataset,labels,k):
    '''
    :param inX: 输入测试样本
    :param dataset: 训练样本集
    :param labels: 训练样本标签
    :param k: top k 最近的
    '''
    #shape返回矩阵的行列数
    #shape[0]表获取数据集的行数，也就是样本的数量
    dataSetSize = dataset.shape[0]
    '''
    下面的求距离过程按照欧式距离的公式计算，即根号(x^2 + y^2)
    '''
    #比如inX = [0,1],
    #                 tile(inX,(4,1))=[[0.0,1.0],
    #                                 [0.0,1.0],
    #                                 [0.0,1.0],
    #                                 [0.0,1.0],]
    difMat = np.tile(inX,(dataSetSize,1)) - dataset
    #difMat是输入样本与每个训练样本的差值，然后对每个差值进行平方运算
    #difMat是一个矩阵，矩阵**2表示对矩阵中的每个元素进行平方操作
    sqDiffMat = difMat**2
    sqDistance = sqDiffMat.sum(axis=1)#按行进行累加
    #对平方和进行开根号
    distance = sqDistance**0.5
    #按照升序进行快速排序，返回的是原数组下标，
    #如：x = [30,10,20,40],升序排序后[10,20,30,40],他们的原下标是[1,2,0,3]
    #那么np.argsort(x) = [1,2,0,3]
    sortedDistIndicies = np.argsort(distance)
    #存放最终的分类结果及相应的结果投票数
    classCount = {}
    #投票过程
    for i in range(k):
        print(sortedDistIndicies[i])
        #第i个最近样本下标对应的分类结果
        voteLabel = labels[sortedDistIndicies[i]]

        #将投票数加1
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    #把分类结果进行排序，返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(fileName):
    '''
    从文件中读入数据，并存储为矩阵
    '''
    fr = open(fileName)
    numberOfLines = len(fr.readlines())#获取样本的行数
    returnMat =np.zeros([numberOfLines,3])#创建一个二维矩阵，nx3
    classLabelVector = []   #存放训练样本标签
    fr = open(fileName)
    index = 0
    for line in fr.readlines():
        #把回车符号去除
        line = line.strip()
        #把每一行数据用\t分割
        listFromLine = line.split('\t')
        print(index,listFromLine)
        #把分割好的数据放入数据集，index为该样本数据的下标，即放到第几行
        returnMat[index,:] = listFromLine[0:3]

        # 把该样本对应的标签放至标签集，顺序与样本集对应。
        classLabelVector.append(int(listFromLine[-1]))
        index = index+1
    return returnMat, classLabelVector
def autoNorm(dataSet):
    """    训练数据归一化    """
    # 获取数据集中每一列的最小数值
    # 以createDataSet()中的数据为例，group.min(0)=[0,0]
    minVals = dataSet.min(0)
    # 获取数据集中每一列的最大数值
    # group.max(0)=[1, 1.1]
    maxVals = dataSet.max(0)
    # 最大值与最小的差值
    ranges = maxVals - minVals
    # 创建一个与dataSet同shape的全0矩阵，用于存放归一化后的数据
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    # 把最小值扩充为与dataSet同shape，然后作差，具体tile请翻看 第三节 代码中的tile
    normDataSet = dataSet - np.tile(minVals, (m,1))
    # 把最大最小差值扩充为dataSet同shape，然后作商，是指对应元素进行除法运算，而不是矩阵除法。
    # 矩阵除法在numpy中要用linalg.solve(A,B)
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals
def datingClassTest():
    # 将数据集中10%的数据留作测试用，其余的90%用于训练
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    print('data',datingDataMat)
    print('label long',datingLabels)
    #load data setfrom file
   # normMat, ranges, minVals = autoNorm(datingDataMat)
    m = datingDataMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(datingDataMat[i,:],datingDataMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d, result is :%s" % (classifierResult, \
            datingLabels[i],classifierResult==datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)
if __name__=="__main__":
    datingClassTest()


