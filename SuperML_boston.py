import mglearn
import numpy as np  
import pandas as pd
import sklearn 
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import mglearn
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
boston = load_boston()
print("데이터의 형태: {}".format(boston.data.shape))

#데이터의 구조 파악
print("structure of data :{}".format(boston.keys()))
print("structure of data :{}".format(boston['DESCR']))

#데이터 셋 불러오기

X,y = mglearn.datasets.load_extended_boston()

print("X.shape :{}".format(X.shape))

#최근접 이웃 분류
mglearn.plots.plot_knn_classification(n_neighbors=1)

#테스트 세트 성능 평가
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
   cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# 1에서 10까지 n_neighbors를 적용
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train, y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="훈련 정확도")
plt.plot(neighbors_settings, test_accuracy, label="테스트 정확도")
plt.ylabel("정확도")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

