import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def standardization(model):
    scaler = StandardScaler()
    scaler.fit(model)
    STD_data = scaler.transform(model)
    return STD_data


def train(X, Y):
    KNN = KNeighborsClassifier(n_neighbors=9)
    KNN.fit(X, Y)
    return KNN


def testing(KNN, X, Y):
    predicted = KNN.predict(X)
    accuracy = classification_report(Y, predicted)
    return accuracy


CLASS_1_MEAN = np.array([2.5, 0.25])
CLASS_1_COVARIANCE = np.array([[1, 0.75], [0.5, 1]])
CLASS_2_MEAN = np.array([-1.5, 0])
CLASS_2_COVARIANCE = np.array([[1, 0.5], [0.5, 1.5]])

# creating models
C1_models = np.random.multivariate_normal(
    CLASS_1_MEAN, CLASS_1_COVARIANCE, 100)
C2_models = np.random.multivariate_normal(
    CLASS_2_MEAN, CLASS_2_COVARIANCE, 100)
C1_lable = np.zeros((100, 1), dtype='int')
C2_lable = np.ones((100, 1), dtype='int')

# creating whole data lables in one 370x1 array
labels = np.vstack((C1_lable, C2_lable))

# normalizing data
features = np.vstack((C1_models, C2_models))
standard_feature = standardization(features)

# adding class lable to data
data = np.append(features, labels, axis=1)

# learn percent = 70% and test = 30% -->
X_train, X_test, y_train, y_test = train_test_split(
    data[:, :-1], data[:, -1], test_size=0.3, random_state=4)

# training classifre and test it
knn = train(X_train, y_train)
accuracy = testing(knn, X_test, y_test)
print(accuracy)
