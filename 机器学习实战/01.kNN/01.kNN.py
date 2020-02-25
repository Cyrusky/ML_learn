import numpy as np
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # 绘图库
import operator

def createDataSet():
    group = np.array([
        [1.0,1.1],
        [1.0,1.0],
        [0,0],
        [0,0.1]
    ])
    labels = ['A','A','B','B']
    Xa, Ya, Xb, Yb = [],[],[],[]
    for index in range(len(labels)):
        print(group[index])
        if labels[index] is 'A':
            Xa.append(group[index][0])
            Ya.append(group[index][1])
        else:
            Xb.append(group[index][0])
            Yb.append(group[index][1])
    print(Xa,Xb,Ya,Yb)
    # plt.scatter(Xa, Ya, color='blue', label='A', alpha=0.6)
    # plt.scatter(Xb, Yb, color='green', label='B', alpha=0.6)
    # plt.legend(loc=(1,1))
    # plt.show()
    return group, labels

def distance(inX, inY):
    return ((inY[0] - inX[0])** 2 +(inY[1] - inX[1])** 2 ) ** 0.5

def classfy0(inX, dataSet, labels,k):
    """
    K近邻算法实现
    Arguments:
        inX {array} -- 需要分类的数据
        dataSet {array or ndArrary} -- 所有数据集
        labels {array} -- 所有标签
        k {int} -- k近邻中的K
        具体的公式为:
        https://www.zhihu.com/equation?tex=distance=\sqrt{(xA_0-xB_0)^2%2B(xA_1-xB_1)^2}
    """
    dist = []
    for i in range(dataSet.shape[0]):
        dist.append({
            "data": dataSet[i],
            "distance": distance(inX, dataSet[i]),
            "classify": labels[i]
        })
    # print(dist)
    sortedClassCount = sorted(dist, key=lambda x: x.get('distance', 0), reverse=False)
    # print(json.dumps(list(sortedClassCount)))
    # 计算 k 个近邻分类投票
    classes = {
        'A': 0,
        'B': 0
    }
    for i in range(k):
        if sortedClassCount[k].get('classify', None) is 'A':
            classes['A'] = classes['A'] + 1
        elif sortedClassCount[k].get('classify', None) is 'B':
            classes['B'] = classes['B'] + 1
    group = np.array([[x for x in dataSet]+[inX]])
    classify = 'A' if classes['A'] > classes['B'] else 'B'
    labels.append(classify)
    return group, labels, classify

if __name__ == "__main__":
    group, labels = createDataSet()
    print(classfy0([0,0], group, labels, 3))
