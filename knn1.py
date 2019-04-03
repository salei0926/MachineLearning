import operator
import numpy as np
def createDataSet():
    group = np.array([[1.0,1.1],
                      [1.0,1.0],
                      [0.,0.],
                      [0.,0.1]
                      ])
    labels = ['A','A','B','B']
    return group,labels
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
        #第i个最近样本下标对应的分类结果
        voteLabel = labels[sortedDistIndicies[i]]
        #将投票数加1
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    #把分类结果进行排序，返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__=='__main__':
    #导入数据
    dataset,labels = createDataSet()
    inX = [2,2]
    className = classify0(inX,dataset,labels,3)
    print('分类结果是：',className)