import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd 
import mglearn
import sklearn
import scipy as sp
from sklearn.datasets import load_iris
iris_dataset = load_iris()
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train, y_test = train_test_split(
    iris_dataset['data'],iris_dataset['target'],random_state=0)

#train 데이터 셋의 크기 확인!!
print("X_train 크기:{}".format(X_train.shape))
print("y_train 크기:{}".format(y_train.shape))
#test 데이터 셋의 크기 확인!!
print("X_test 크기:{}".format(X_test.shape))
print("y_test 크기:{}".format(y_test.shape))

print(X_train)



#dataset 의 산점도 그리고 직관적으로 데이터의 형태 판단!!
iris_dataframe = pd.DataFrame(X_train,columns = iris_dataset.feature_names)
iris_test=pd.plotting.scatter_matrix(iris_dataframe,c=y_train , figsize=(15,15),marker='0',
                           hist_kwds={'bins':20},s=60,alpha=.8 , cmap = mglearn.cm3)
#항상 matplot show 놓치면 안됨.
plt.show()