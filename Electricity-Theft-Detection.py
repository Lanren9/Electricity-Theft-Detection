import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils.np_utils import *
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from keras.layers import LSTM
from keras import regularizers
from matplotlib import pyplot
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sna
import pickle
from matplotlib.axis import Axis
from tensorflow.keras.utils import plot_model
from keras.models import Model
from keras.layers import add,Input,Activation,Dense

#读取数据集
data = pd.read_csv("/home/zhaozhiqiang/balance_data.csv")
data.drop(['Unnamed: 0'],axis=1,inplace=True)
y=data['label']
data.drop(['label'],axis=1,inplace=True)

x=np.array(data)
y=np.array(y)
print("已加载完成数据集")

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20) #, random_state=42
x_train = x_train.reshape(x_train.shape[0], 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 1)

Y_test = to_categorical(y_test,2)
Y_train = to_categorical(y_train,2)


def ETD(train_x,train_y,test_x,test_y):   
    train_input=Input(shape=(48,1))
    train_output=Dense(10, activation='tanh',input_dim=3,use_bias=True)(train_input)
    train_output=LSTM(300,return_sequences=True,kernel_regularizer=regularizers.l2(0.01))(train_output)
    train_output=LSTM(300,kernel_regularizer=regularizers.l2(0.01))(train_output)
    train_output=Dense(2,activation='softmax')(train_output)
    model=Model(inputs=train_input,outputs=train_output)
    #查看网络结构
    model.summary()
    #保存模型结构图片
    plot_model(model, to_file='.\model.png')
    #编译模型
    model.compile(optimizer='RMSprop',loss='categorical_crossentropy',metrics=['accuracy'])
    #训练模型
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3,factor=0.1,verbose=1,min_lr=0,mode='auto',epsilon=0.001,cooldown=0)
    EarlyStop=EarlyStopping(monitor='val_accuracy',min_delta=0.001,patience=5,verbose=1, mode='auto')
    history=model.fit(train_x,train_y,batch_size=512,epochs=1,verbose=1,callbacks=[reduce_lr,EarlyStop],validation_data=(test_x,test_y))
    #评估模型
    _, train_acc = model.evaluate(train_x, train_y,verbose=0,batch_size=512)
    _, test_acc = model.evaluate(test_x, test_y, verbose=0,batch_size=512)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    #生成混淆矩阵
    y_pred = model.predict(test_x)
    y_pred = tf.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    return cm,history

# Loss and accuracy figures
def drawing(history):
    fig1 = pyplot.figure('Figure1',figsize = (8,6))
    pyplot.xticks(fontsize=11)
    pyplot.yticks(fontsize=11)
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend(loc = 'lower right',fontsize=15)
    pyplot.ylabel('Accuracy',fontsize=14)
    pyplot.xlabel('Epochs',fontsize=14)
    pyplot.savefig('acc.pdf')
    pyplot.legend(loc = 'lower right',fontsize=15)
    fig2 = pyplot.figure('Figure2',figsize = (8,6))
    pyplot.xticks(fontsize=11)
    pyplot.yticks(fontsize=11)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend(loc = 'upper right',fontsize=15)
    pyplot.ylabel('Loss',fontsize=14)
    pyplot.xlabel('Epochs',fontsize=14)
    pyplot.savefig('loss.pdf')
    pyplot.legend(loc = 'upper right',fontsize=15)



def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=pyplot.cm.Blues):
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title,fontsize=20)
    pyplot.colorbar()
    tick_marks = np.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45,fontsize=15)
    pyplot.yticks(tick_marks, classes,fontsize=15)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pyplot.ylabel('True label',fontsize=16)
    pyplot.xlabel('Predicted label',fontsize=16)
    pyplot.tight_layout()



#main()
if __name__ == "__main__":
    cm,history=ETD(x_train,Y_train,x_test,Y_test)
    print(cm)
    drawing(history)
    pyplot.figure(figsize = (8,8))
    sna.set(font_scale=1.5)
    pyplot.grid(visible=False)
    plot_confusion_matrix(cm, classes=['Normal','Theft'],title='Confusion matrix')
    pyplot.savefig("confusion matrix.pdf")
