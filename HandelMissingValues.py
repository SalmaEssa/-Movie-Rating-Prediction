import pandas as pd
import json

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import Lasso
from DataCleaner import *
import os.path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

def regression_without_pca(X_train,X_test,y_train,y_test,features):

    poly_features = PolynomialFeatures(degree=2)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.fit_transform(X_test)

    #poly Regression

    filename = 'poly_reg.sav'
    if os.path.exists(filename):
        model_poly = pickle.load(open(filename, 'rb'))
    else:
        model_poly = linear_model.LinearRegression().fit(X_train_poly, y_train)
        pickle.dump(model_poly, open(filename, 'wb'))

    poly_predicted = model_poly.predict(X_test_poly)
    poly_score = model_poly.score(X_test_poly, y_test)

    #lin Regression
    filename = 'linear_reg.sav'
    if os.path.exists(filename):
        model_linear = pickle.load(open(filename, 'rb'))
    else:
        model_linear =  linear_model.LinearRegression().fit(X_train, y_train)
        pickle.dump(model_linear, open(filename, 'wb'))

    prediction = model_linear.predict(X_test)

    ##Ridge Regression
    filename = 'ridge_reg.sav'
    if os.path.exists(filename):
        model_ridge = pickle.load(open(filename, 'rb'))
    else:
        model_ridge =  Ridge(alpha = 0.01).fit(X_train, y_train) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, co
        pickle.dump(model_ridge, open(filename, 'wb'))

    Ridge_test_score = model_ridge.score(X_test, y_test)
    pred = model_ridge.predict(X_test)
    mse_ridge = np.mean((pred - y_test)**2)

    #Lasso Regression

    filename = 'lasso_reg.sav'
    if os.path.exists(filename):
        model_lasso = pickle.load(open(filename, 'rb'))
    else:
        model_lasso =  Lasso(alpha = 0.3).fit(X_train,  y_train)
        pickle.dump(model_lasso, open(filename, 'wb'))

    pred_lasso = model_lasso.predict(X_test)
    mse_lasso = np.mean((pred_lasso - y_test)**2)
    lasso_score = model_lasso.score(X_test,y_test)

    for i in features:
        plt.scatter(X_train[i],y_train)

        X1 = np.min(X_test[i])
        X1List = np.where(X_test[i] == X1)[0]
        X1Index = X1List[0]
        Y1 = prediction[X1Index]

        X2 = np.max(X_test[i])
        X2List = np.where(X_test[i] == X2)[0]
        X2Index = X2List[0]
        Y2 = prediction[X2Index]
        XX = [X1, X2]
        YY = [Y1, Y2]
        plt.xlabel(i)
        plt.ylabel('rate')
        plt.plot(XX,YY,'red')

        #plt.show()
    print('Mean Square Error Of Linear', metrics.mean_squared_error(y_test, prediction))
    print('Score Of Linear', model_linear.score(X_test, y_test))
    print('--------------------------')
    print('Mean Square Error Of Ploy', metrics.mean_squared_error(y_test, poly_predicted))
    print('Score Of Ploy',poly_score )
    print('--------------------------')
    print('Mean Square Error Of Ridge Regression ', mse_ridge)
    print('Score Of Ridge Regression', Ridge_test_score)
    print('--------------------------')
    print('Mean Square Error Of Lasso Regression ', mse_lasso)
    print('Score Of Lasso Regression', lasso_score)
def regression_with_pca(X_train,X_test,y_train,y_test,features):
    pass


