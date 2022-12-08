import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


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


CLASS_1_MEAN = np.array([2.5, 0.25])
CLASS_1_COVARIANCE = np.array([[1, 0.75], [0.5, 1]])
CLASS_2_MEAN = np.array([-1.5, 0])
CLASS_2_COVARIANCE = np.array([[1, 0.5], [0.5, 1.5]])

# creating models
C1_models = np.random.multivariate_normal(
    CLASS_1_MEAN, CLASS_1_COVARIANCE, 70)
C1_lable = np.full((70, 1), 0)
C1_models = np.append(C1_models, C1_lable, axis=1)
C2_models = np.random.multivariate_normal(
    CLASS_2_MEAN, CLASS_2_COVARIANCE, 300)
C2_lable = np.full((300, 1), 1)
C2_models = np.append(C2_models, C2_lable, axis=1)

# normalizing datas
wholeData = np.vstack((C1_models, C2_models))

# adding class lable to data
data = np.concatenate((C1_models, C2_models))
data_lable = C1_lable+C2_lable

# learn percent = 70% and test = 30% --> lentgh * 0.3 and lentgh * 0.7 !!!
X_train, X_test, y_train, y_test = train_test_split(
    data, data_lable, test_size=0.3, random_state=4)
print('Train set:', X_train,  y_train)
print('Test set:', X_test,  y_test)

knn = train(trainingData)
f_score = testing(knn, testingData)
print(
    f'F-measure is {f_score:.2f} .')
