import mglearn
import numpy as np  
import pandas as pd
import sklearn 
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import mglearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

#여기서 fig는 그림하나당 크기정하는거고, axes는 그림갯수를 뜻함 
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

#for 문 이해 잘 해야해 zip은 tuple을 하나의 셋트로 묶어주는 것이고 for문안
#에 zip안에 2개의 인자야 LineatSVC,Logistic 하나 axes 둘 model,ax도 두개
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend()


