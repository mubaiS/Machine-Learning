import numpy as np
import matplotlib.pyplot as plt
"""
加载数据
Parameters：
    无
Returns：
    DataMat - 数据列表
    DataLabels - 标签列表
"""

def loadDataSet():
    dataMat = []
    labelMat = []
    f = open("testSet.txt")
    for line in f.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    f.close()
    return dataMat, labelMat

'''
Sigmoid函数
Parameters:
    inX - 数据
Returns：
    Sigmoid函数
'''
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

'''
梯度上升算法
Parameters：
    dataMatIn - 数据集
    classLabels - 数据标签
Returns：
    weights.getA() - 求得的权重数组
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alaph = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alaph * dataMatrix.transpose() * error
    return weights.getA()

'''
绘制数据集
Parameters：
    weights - 权重参数
Returns：
    无
'''
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()                                    #加载数据集
    dataArr = np.array(dataMat)                                            #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()

if __name__ == "__main__":
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)