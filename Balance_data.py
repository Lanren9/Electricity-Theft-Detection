import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN

data = pd.read_csv("D:/Experiments/Electricity-Theft-Detection/data_label_final.csv")
data.drop(['Unnamed: 0'],axis=1,inplace=True)
y=data['label']
data.drop(['label'],axis=1,inplace=True)

x=np.array(data)
y=np.array(y)

ana = ADASYN(sampling_strategy=1,random_state=0)
x_ana, y_ana = ana.fit_resample(x, y)
X = pd.DataFrame(x_ana)
Y = pd.DataFrame(y_ana)
X['label']=y_ana

X.to_csv('D:/Experiments/Electricity-Theft-Detection/balance_data.csv')
