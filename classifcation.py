import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import  svm
from sklearn.multiclass import OneVsRestClassifier
from DataCleaner import *
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def SVM(X_train,X_test,y_train,y_test):
    filenames=['SVM1.sav','SVM2.sav','SVM3.sav','SVM4.sav','SVM5.sav','SVM7.sav','SVM6.sav','SVM8.sav','SVM9.sav']
    c=[1,1000,1000000]
    ker=['linear' , 'poly' , 'rbf']
    training_time , testing_time , max_accuracy=0,0,0
    for i in range(len(c)):
        for j in range(len(ker)):
            k=i*len(c)+j
            filename = filenames[k]
            if os.path.exists(filename):
                svm = pickle.load(open(filename, 'rb'))
            else:
                t1 = time.time()
                svm = OneVsRestClassifier(SVC(kernel=ker[j], C=c[i])).fit(X_train, y_train)
                t2 = time.time()
                training_time=max(training_time, t2-t1)
                pickle.dump(svm, open(filename, 'wb'))
            t1 = time.time()
            accuracy = svm.score(X_test, y_test)*100
            t2 = time.time()
            testing_time = max(testing_time, t2-t1)
            max_accuracy = max(accuracy,max_accuracy)
            print('One VS Rest SVM accuracy with kernel={} and c={} is : {}%'.format(ker[j], c[i],  accuracy))
    return training_time,training_time,max_accuracy

def KNN(X_train,X_test,y_train,y_test):
    weight=['uniform','distance']
    filenames=[]
    for i in range(40):
        filenames.append('KNN'+str(i+1)+'.sav')
    training_time, testing_time, max_accuracy = 0, 0, 0
    for j in range(len(weight)):
        for i in tqdm(range(1, 20)):
            k = j * len(weight) + i
            filename = filenames[k]
            if os.path.exists(filename):
                knn = pickle.load(open(filename, 'rb'))
            else:
                t1 = time.time()
                knn = KNeighborsClassifier(n_neighbors=i , weights=weight[j])
                knn.fit(X_train, y_train)
                t2 = time.time()
                training_time = max(training_time , t2-t1)
                pickle.dump(knn, open(filename, 'wb'))
            t1 = time.time()
            pred_i = knn.predict(X_test)
            accuracy = np.mean(pred_i == y_test) * 100
            t2 = time.time()
            testing_time = max(testing_time, t2 - t1)
            max_accuracy = max(accuracy, max_accuracy)
            print('KNN accuracy: with n_neighbors={} and  weights={} is: {}%'.format(i ,weight[j] ,accuracy))
    return training_time, training_time, max_accuracy


def Logistic_Regression(X_train,X_test,y_train,y_test):
    c = [1, 1000, 1000000]
    filenames = ['Logistic_Regression1.sav', 'Logistic_Regression2.sav', 'Logistic_Regression3.sav']
    training_time, testing_time, max_accuracy = 0, 0, 0
    for i in range(len(c)):
        filename = filenames[i]
        if os.path.exists(filename):
            logistic_regression_model = pickle.load(open(filename, 'rb'))
        else:
            t1 = time.time()
            logistic_regression_model = LogisticRegression(C=c[i]).fit(X_train, y_train)
            t2 = time.time()
            training_time = max(training_time, t2 - t1)
            pickle.dump(logistic_regression_model, open(filename, 'wb'))
        t1 = time.time()
        accuracy = logistic_regression_model.score(X_test, y_test) * 100
        t2 = time.time()
        testing_time = max(testing_time, t2 - t1)
        max_accuracy = max(max_accuracy, accuracy)
        print('Logistic Regression accuracy: ' + str(accuracy))
    return training_time, training_time, max_accuracy

def Decision_Tree(X_train,X_test,y_train,y_test):
    max_feature=['sqrt', 'log2', None]
    depth=[5,10, 20 , 50,None]
    filenames=['DT1.sav' , 'DT2.sav','DT3.sav','DT4.sav','DT5.sav','DT6.sav','DT7.sav','DT8.sav','DT9.sav','DT10.sav','DT11.sav','DT12.sav']
    training_time, testing_time, max_accuracy = 0, 0, 0
    for i in range(len(max_feature)):
        for j in range(len(depth)):
            k = i * len(max_feature) + j
            filename = filenames[k]
            if os.path.exists(filename):
                decision_tree_model = pickle.load(open(filename, 'rb'))
            else:
                t1=time.time()
                decision_tree_model = DecisionTreeClassifier(max_features=max_feature[i] , max_depth=depth[j]).fit(X_train, y_train)
                t2 = time.time()
                training_time=max(training_time , t2-t1)
                pickle.dump(decision_tree_model, open(filename, 'wb'))
            t1 = time.time()
            accuracy = decision_tree_model.score(X_test, y_test)*100
            t2 = time.time()
            testing_time = max(testing_time, t2 - t1)
            max_accuracy=max(max_accuracy ,accuracy)
            print('Decision Tree accuracy with max_features={} and max_depth={} is {}'.format(max_feature[i], depth[j], accuracy))
    return training_time, testing_time, max_accuracy

def classification_withoutPCA(X_train,X_test,y_train,y_test):
    svm_triningTime, svm_testingTime, svm_accuracy = SVM(X_train,X_test,y_train,y_test)
    knn_triningTime, knn_testingTime, knn_accuracy = KNN(X_train,X_test,y_train,y_test)
    LR_triningTime, LR_testingTime, LR_accuracy = Logistic_Regression(X_train,X_test,y_train,y_test)
    DT_triningTime, DT_testingTime, DT_accuracy = Decision_Tree(X_train,X_test,y_train,y_test)
    model_name = ('SVM', 'KNN', 'Logistic Regression', 'Decision Tree')
    y_pos = np.arange(len(model_name))
    training_time = [svm_triningTime, knn_triningTime, LR_triningTime, DT_triningTime]
    testing_time = [svm_testingTime, knn_testingTime, LR_testingTime, DT_testingTime]
    accuracy = [svm_accuracy, knn_accuracy, LR_accuracy, DT_accuracy]

    plt.bar(y_pos, training_time, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('time')
    plt.title('Training Time')
    plt.show()

    plt.bar(y_pos, testing_time, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('time')
    plt.title('Testing Time')
    plt.show()

    plt.bar(y_pos, accuracy, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('accuracy')
    plt.title('Acuuracy')
    plt.show()


def classification_with_PCA(X_train,X_test,y_train,y_test):
    pca = PCA(n_components=200)
    principalComponents = pca.fit_transform(X_train)
    principalComponents2 = pca.transform(X_test)
    # print(pca.explained_variance_ratio_)
    X_train = pd.DataFrame(data=principalComponents
                           , columns=['pca'] * 200)
    X_test = pd.DataFrame(data=principalComponents2
                          , columns=['pca'] * 200)
    svm_triningTime, svm_testingTime, svm_accuracy = SVM(X_train,X_test,y_train,y_test)
    knn_triningTime, knn_testingTime, knn_accuracy = KNN(X_train,X_test,y_train,y_test)
    LR_triningTime, LR_testingTime, LR_accuracy = Logistic_Regression(X_train,X_test,y_train,y_test)
    DT_triningTime, DT_testingTime, DT_accuracy = Decision_Tree(X_train,X_test,y_train,y_test)
    model_name = ('SVM', 'KNN', 'Logistic Regression', 'Decision Tree')
    y_pos = np.arange(len(model_name))
    training_time = [svm_triningTime, knn_triningTime, LR_triningTime, DT_triningTime]
    testing_time = [svm_testingTime, knn_testingTime, LR_testingTime, DT_testingTime]
    accuracy = [svm_accuracy, knn_accuracy, LR_accuracy, DT_accuracy]

    plt.bar(y_pos, training_time, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('time')
    plt.title('Training Time')
    plt.show()

    plt.bar(y_pos, testing_time, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('time')
    plt.title('Testing Time')
    plt.show()

    plt.bar(y_pos, accuracy, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('accuracy')
    plt.title('Acuuracy')
    plt.show()




