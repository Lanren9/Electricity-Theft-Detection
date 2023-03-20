import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from sktime.datasets import load_arrow_head  # univariate dataset
from sktime.datasets import load_basic_motions  # multivariate dataset
from sktime.transformations.panel.rocket import Rocket

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
x_train, y_train = load_arrow_head(split="train", return_X_y=True)

rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
rocket.fit(x_train)
x_train_transform = rocket.transform(x_train)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(x_train_transform, y_train)

x_test, y_test = load_arrow_head(split="test", return_X_y=True)
x_test_transform = rocket.transform(x_test)


a=classifier.score(x_test_transform, y_test)

print("score:",a)
