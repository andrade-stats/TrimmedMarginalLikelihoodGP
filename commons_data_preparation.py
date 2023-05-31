import sklearn.preprocessing
import numpy


def scale_X(X_train, X_test = numpy.zeros((0,0))):
    dataScalerX = sklearn.preprocessing.RobustScaler().fit(X_train)
    X_train = dataScalerX.transform(X_train)
    if X_test.shape[0] > 0:
        X_test = dataScalerX.transform(X_test)
    
    return X_train, X_test, dataScalerX


def scale_y(y_train, y_test = numpy.zeros((0))):
    dataScalerY = sklearn.preprocessing.RobustScaler().fit(numpy.reshape(y_train, (-1,1)))
    y_train = dataScalerY.transform(numpy.reshape(y_train, (-1,1)))[:,0]
    if y_test.shape[0] > 0:
        y_test = dataScalerY.transform(numpy.reshape(y_test, (-1,1)))[:,0]
    
    return y_train, y_test, dataScalerY
