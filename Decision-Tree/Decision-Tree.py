from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    with open('lenses.txt', 'r') as fr:  # 加载文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 处理文件
    lenses_target = []  # 提取每组数据的类别，保存在列表里
    for each in lenses:
        lenses_target.append(each[-1])
    # print(lenses_target)

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 特征标签
    lenses_list = []  # 保存lenses数据的临时列表
    lenses_dict = {}  # 保存lenses数据的字典，用于生成pandas
    for each_label in lensesLabels:  # 提取信息，生成字典
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)                                                        #打印字典信息
    lenses_pd = pd.DataFrame(lenses_dict)  # 生成pandas.DataFrame
    # print(lenses_pd)                                                        #打印pandas.DataFrame
    le = LabelEncoder()  # 创建LabelEncoder()对象，用于序列化
    for col in lenses_pd.columns:  # 序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # print(lenses_pd)                                                        #打印编码信息

    clf = tree.DecisionTreeClassifier(max_depth=4)  # 创建DecisionTreeClassifier()类
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)  # 使用数据，构建决策树
    print(clf.predict([[1, 1, 1, 0]]))  # 预测