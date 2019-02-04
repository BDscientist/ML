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


mglearn.plots.plot_linear_regression_wave()
#plt.show()

#선형회귀(최소제곱법)
from sklearn.linear_model import LinearRegression
X,y  = mglearn.datasets.make_wave(n_samples=60)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
lr = LinearRegression().fit(X_train,y_train)

print("lr.coef_:{}".format(lr.coef_))
print("lr.intercept_:{}".format(lr.intercept_))

#R^2값
print("훈련세트 점수:{:.4f}".format(lr.score(X_train,y_train)))
print("훈련세트 점수:{:.4f}".format(lr.score(X_test,y_test)))

#ridge regression
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge10.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge01.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge01.score(X_test, y_test)))

#alpha 값에 따른 그래프의 변화
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("계수 목록")
plt.ylabel("계수 크기")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()
#boston 주택가격의 회귀 perfomance
from sklearn.datasets import load_boston
boston = load_boston()
boston_reg = mglearn.plots.plot_ridge_n_samples()
plt.show(boston_reg)
