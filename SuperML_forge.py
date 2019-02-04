import mglearn
import numpy as np  
import pandas as pd
import sklearn 
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib


# 데이터셋을 만듭니다.
X, y = mglearn.datasets.make_forge()

# 산점도를 그립니다.
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번째 특성")
plt.show()
print("X.shape: {}".format(X.shape))
print("==================================")
#최근접 이웃 분류
mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)

#forge 데이터 셋을 train,test로 나눕니다.
from sklearn.model_selection import train_test_split
X,y  = mglearn.datasets.make_forge()

X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)
#k-nn 알고리즘 적용
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print("test 세트 예측:{}".format(clf.predict(X_test)))
print("====================================")
print("테스트 세트 정확도:{:.2f}".format(clf.score(X_test,y_test)))
print("=================================")

#knn 알고리즘 변화도 그래프로 그리기
fig, axes  = plt.subplots(1,3,figsize=(10,3))

for n_neighbors ,ax in zip([1,3,9],axes):
    #fit 메서드는 self 객체를 반환함.
    #그래서 객체생성과 fit 메서드를 한 줄에 쓸수 있음.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True , eps=0.5 ,ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{}이웃".format(n_neighbors))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 0")
axes[0].legend(loc=3)

#wave에 대한 데이터셋그래프
X,y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("특성")
plt.ylabel("타깃")
plt.show()
#k-최근접 이웃 회귀
mglearn.plots.plot_knn_regression(n_neighbors=1)

#cancer에 대한 데이터셋
from sklearn.datasets import load_breast_cancer
cancer  = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

#DATA의 Structure(중요!)
print("cancer 데이터의 특징: {}".format(cancer['DESCR']))
print("암의 형태 : {}".format(cancer.data.shape))

#cancer data중 0으로 진단받은 열만 뽑기 코드 (builtin code!!)
result = filter(lambda i: i==0, cancer['target'])
#print(result.bincount)
print(list(result))

#class별 샘플 개수(파이썬 스러운 코드)
print("클래스별 샘플 갯수:\n :{}".format(
    {n:v for n,v  in zip(cancer.target_names,np.bincount(cancer.target))}))

print("특성이름 :\n{}".format(cancer.feature_names))


