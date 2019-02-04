import sklearn 
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import mglearn
import numpy as np
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso



#데이터 셋 불러오기
boston = load_boston()
X,y = mglearn.datasets.load_extended_boston()
#데이터 나누기 TRAIN, TEST
X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)
lasso = Lasso().fit(X_train , y_train)
print("훈련세트 점수:{:.2f}".format(lasso.score(X_train,y_train)))
print("테스트 세트 점수:{:.2f}".format(lasso.score(X_test,y_test)))
print("사용한 특성의 수:{}".format(np.sum(lasso.coef_ !=0)))

#alpha값을 늘려보자. 데이터가 high density ---> alpha의 의미
# "max_iter" 기본값을 증가시키지 않으면 max_iter 값을 늘리라는 경고가 발생합니다.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso001.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso001.score(X_test, y_test)))
print("사용한 특성의 수: {}".format(np.sum(lasso001.coef_ != 0)))

#alpha값 0.00001
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("사용한 특성의 수: {}".format(np.sum(lasso00001.coef_ != 0)))

#alpha값에따른 데이터 성능의 평가


from sklearn.linear_model import Ridge

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("계수 목록")
plt.ylabel("계수 크기")
plt.show()