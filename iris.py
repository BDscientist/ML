from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("iris_dataset의 키: \n".format(iris_dataset.keys()))

print("=============================================")

#DESCR은 데이터 셋에 대한 간략한 설명이 들어 있습니다.
print(iris_dataset['DESCR'][:193]+"\n...")

#target_names 의 값은 우리가 예측하려는 붓꽃 품종의 읾을 문자열 배열을 가지고 있음
print("타겟의 이름 :{}".format(iris_dataset['target_names']))

#feature_names의 값은 특성을 설명하는 문자열 리스트입니다.
print("특성의 이름:\n{}".format(iris_dataset['feature_names']))

#실제 데이터는 target과 data 필드에 들어 있습니다. data는 꽃잎의 길이와 폭,
#꽃받침의 길이와 폭을 수치값으로 가지고 있는 numpy 배열입니다.
print("data 타입:{}".format(type(iris_dataset['data'])))

#data 배열의 행은 개개으 꽃이 되며 열은 각꽃에서 구한 네개의 측정치입니다.
print("data의 크기:{}".format(iris_dataset['data'].shape))

print("data의 처음 다섯행 :{}".format(iris_dataset['data'][:5]))

#data의 target 배열도 샘플 붓꽃의 품종을 담은 numpy 배열입니다.
print("target의 타입:{}".format(type(iris_dataset['target'])))
print("target의 크기:{}".format(type(iris_dataset['target'].shape)))

#데이터의표현  0:setosa , 1:versicolor , 2:virginica
print("타깃:\n".format(iris_dataset['target']))


