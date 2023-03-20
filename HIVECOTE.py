import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sktime.classification.hybrid import HIVECOTEV2
from sktime.datasets import load_unit_test


data = pd.read_csv("/home/zhaozhiqiang/data_label_final.csv")
data.drop(['Unnamed: 0'],axis=1,inplace=True)
y=data['label']
data.drop(['label'],axis=1,inplace=True)
x=np.array(data)
y=np.array(y)
print("已加载完成数据集")
print('开始平衡数据集')
# 平衡数据集
ana = ADASYN(sampling_strategy=1,random_state=42)
x_ana, y_ana = ana.fit_resample(x, y)


print("开始划分数据集")

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x_ana, y_ana, test_size=0.20) #, random_state=42
x_train, y_train = load_unit_test(split="train", return_X_y=True)
x_test, y_test = load_unit_test(split="test", return_X_y=True)
clf = HIVECOTEV2()
clf.fit(x_train, y_train)
a=clf.score(x_test,y_test)
print("score:",a)
