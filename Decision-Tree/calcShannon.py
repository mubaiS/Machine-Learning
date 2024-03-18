from math import log


"""
创建测试数据集
Parameters：
    无
Returns：
    dataSet - 数据集
    labels - 分类属性
"""
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['放贷','不放贷']
    return dataSet, labels


"""
计算经验熵
Parameters：
    dataSet - 数据集
Returns：
    shannonEnt - 经验熵
"""
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        currLabels = featVec[-1]
        if currLabels not in labelCount.keys():
            labelCount[currLabels] = 0
        labelCount[currLabels] += 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


"""
按照给定特征划分数据集
Parameters：
    dataSet - 数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
Returns：
    retDataSet - 处理后的数据集
"""

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
选择最优特征
Parameters:
    dataSet - 数据集
Returns：
    bestFeature - 信息增益最大的特征的索引值
'''

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回信息增益最大的特征的索引值


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    # print(dataSet)
    # print(calcShannonEnt(dataSet))
    print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))