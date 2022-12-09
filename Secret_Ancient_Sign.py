import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train(X, Y):
    classifier = svm.SVC(kernel='rbf')
    classifier.fit(X, Y)
    return classifier


def testing(clf, X, Y):
    predicted = clf.predict(X)
    accuracy = classification_report(Y, predicted)
    return accuracy


CLASS_1_MEAN = np.array([2.5, 0.25])
CLASS_1_COVARIANCE = np.array([[1, 0.75], [0.5, 1]])
CLASS_2_MEAN = np.array([-1.5, 0])
CLASS_2_COVARIANCE = np.array([[1, 0.5], [0.5, 1.5]])

# creating models
C1_models = np.random.multivariate_normal(
    CLASS_1_MEAN, CLASS_1_COVARIANCE, 70)
C2_models = np.random.multivariate_normal(
    CLASS_2_MEAN, CLASS_2_COVARIANCE, 300)
C1_lable = np.zeros(70, dtype='int')
C2_lable = np.ones(300, dtype='int')

# creating whole data lables in one 370, array
labels = np.append(C1_lable, C2_lable)

# creating whole data feature in one 370x2 array
features = np.vstack((C1_models, C2_models))

# learn percent = 70% and test = 30% -->
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=4)

# training classifre and test it
classifier = train(X_train, y_train)
accuracy = testing(classifier, X_test, y_test)
print(accuracy)