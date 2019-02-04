import numpy as np   


X=np.array([[0,1,0,1],[1,0,1,1],[0,0,0,1],[1,0,1,0]])
y=np.array([0,1,0,1])
#axis가 0이면 각행의 순서가 맞는 원소끼리 더해주는것.
#y==0 true flase true false
counts ={}
for label in np.unique(y):
    #클래스마다 반복
    #특성마다 1이 나타난 횟수를 센다.
    counts[label] = X[y == label].sum(axis=0)
print("특성 카운트:\n{}".format(counts))