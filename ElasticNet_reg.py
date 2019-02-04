
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


boston = load_boston()
X,y = mglearn.datasets.load_extended_boston()
#데이터 나누기 TRAIN, TEST
X_train , X_test , y_train , y_test = train_test_split(X,y,random_state=0)
ENreg = ElasticNet(alpha=1,l1_ratio=0.5,normalize=False)
ENreg.fit(X_train , y_train)
pred_cv = ENreg.predict(X_test)

#mse
mse = np.mean((pred_cv - y_test))
print("MSE :{:.2f}".format(mse))
#predict score
ENreg_score=ENreg.score(X_test,y_test)
print("ENreg_score:{:.2f}".format(ENreg_score))