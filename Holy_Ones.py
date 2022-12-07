from numpy.random import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from random import shuffle, seed


def addLable(list_: list, lable: int):
    output = list()
    for thisOne in list_:
        output.append(thisOne+[lable])
    return output


def standardization(model: list):
    scaler = StandardScaler()
    scaler.fit(model)
    STD_Data = scaler.transform(model)
    return STD_Data


def train(data: list):
    KNN = KNeighborsClassifier(n_neighbors=9)
    unsignedData = list(map(lambda arg: arg[0:2], data))
    lable = list(map(lambda arg: arg[2], data))
    KNN.fit(unsignedData, lable)
    return KNN


def testing(KNN, data):
    unsignedData = list(map(lambda arg: arg[0:2], data))
    lable = list(map(lambda arg: arg[2], data))
    predicted = KNN.predict(unsignedData)
    accuracy = metrics.f1_score(lable, predicted, average='binary')
    return accuracy


CLASS_1_MEAN = [2.5, 0.25]
CLASS_1_COVARIANCE = [[1, 0.75], [0.5, 1]]
CLASS_2_MEAN = [-1.5, 0]
CLASS_2_COVARIANCE = [[1, 0.5], [0.5, 1.5]]

# creating models
C1_models = multivariate_normal(
    CLASS_1_MEAN, CLASS_1_COVARIANCE, 70)
C2_models = multivariate_normal(
    CLASS_2_MEAN, CLASS_2_COVARIANCE, 300)

# normalizing datas
C1_models = standardization(C1_models).tolist()
C2_models = standardization(C2_models).tolist()

# adding class lable to datas
C1_models = addLable(C1_models, 0)
C2_models = addLable(C2_models, 1)
data = C1_models+C2_models
seed(12)
shuffle(data)

# learn percent = 70% and test = 30% --> lentgh * 0.3 and lentgh * 0.7 !!!
trainTestBorder = round(len(data)*0.7)
trainingData = data[0:trainTestBorder]
testingData = data[trainTestBorder:]

knn = train(trainingData)
f_score = testing(knn, testingData)
print(
    f'F-measure is {f_score:.2f} .')