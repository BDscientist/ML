import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd 
import mglearn
import sklearn
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()


print(iris_dataset['data'])
print(iris_dataset['target'])

X_train ,X_test , y_train, y_test = train_test_split(
    iris_dataset['data'],iris_dataset['target'],random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
print(knn.fit(X_train,y_train))

#예측하기
#간단한 데이터 셋 꽃받침이 5, 넓이가 2.9 ....
X_new = np.array([[5,2.9,1,0.2]])
print("X_new.shape:{}".format(X_new.shape))
print("=============================")


prediction = knn.predict(X_new)
print("예측 :{}".format(prediction))
print("예측한 타깃의 이름:{}".format(iris_dataset['target_names'][prediction]))
print("==============================")


#모델 평가하기
y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n{}".format(y_pred))
print("테스트 세트의 정확도:{:.2f}".format(np.mean(y_pred == y_test)))
