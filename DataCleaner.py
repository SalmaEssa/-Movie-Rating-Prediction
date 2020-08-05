from datetime import datetime

import pandas as pd
import json
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import json
import numpy as np
from sklearn import preprocessing
from DataCleaner import *
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import os.path
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from tqdm import tqdm

class DataCleaner:
 # type c= classification, type r = reggrision
 # t='t' mean train , t='s' mean test
    def __init__(self,path1,path2,test_file1, test_file2,  Type):
        self.movies = pd.read_csv(path1)
        self.credits = pd.read_csv(path2)
        self.type=Type

        self.test_movies=pd.read_excel(test_file1)
        self.test_credits=pd.read_csv(test_file2)
        self.merge()
        self.merge2()
        self.features = []
        self.actorsDictionary = dict()
        self.directorsDictionary = dict()

    def merge(self):
        self.movies = pd.merge(self.movies, self.credits, left_on='id', right_on='movie_id')
        self.dropDuplicateColumns('t')
        self.dropUnnecessaryColumns('t')
        self.dropMissingRows()
        self.reformat('t')
        if self.type=='c':
            self.calssformat('t')


        train_data =self.movies
        if self.type=='c':
            train_data.to_csv('trainC.csv')
        else:
            train_data.to_csv('trainR.csv')

        #test_data.to_csv('testC.csv')
    #for test files
    def merge2(self):
        self.test_movies = pd.merge(self.test_movies, self.test_credits, left_on='id', right_on='movie_id')
        self.dropDuplicateColumns('s')  #s for test, t for train
        self.dropUnnecessaryColumns('s')
        #self.dropMissingRows()
        self.reformat('s')
        if self.type=='c':
            self.calssformat('s')

        test_data = self.test_movies
        if self.type=='c':
            test_data.to_csv('testC.csv')
        else:
            test_data.to_csv('testR.csv')

    def calssformat(self, t ):
        if t=='t':
            self.movies.loc[self.movies['rate'] == 'High', 'rate'] = 3
            self.movies.loc[self.movies['rate'] == 'Intermediate', 'rate'] = 2
            self.movies.loc[self.movies['rate'] == 'Low', 'rate'] = 1
        else:
            self.test_movies.loc[self.test_movies['rate'] == 'High', 'rate'] = 3
            self.test_movies.loc[self.test_movies['rate'] == 'Intermediate', 'rate'] = 2
            self.test_movies.loc[self.test_movies['rate'] == 'Low', 'rate'] = 1



    def defineCategory(self, i, list, id):
        #bey7ot el feature as coulmn in moviees, then but it =1
        # darama is coulmn, romance is coulumn ...
        for k in list:
            if k[id] in self.movies.columns:
                self.movies.at[i, k[id]] = 1  #1 or 0
            else:
                nColumn = pd.DataFrame({k[id]: [0] * self.movies.shape[0]})
                self.movies = pd.concat([nColumn, self.movies], axis=1)
                self.movies.at[i, k[id]] = 1

    def defineCategories(self):
        for i in range(0, self.movies.shape[0]):
            '''self.movies.at[i, 'release_date']= datetime.strptime(str(self.movies.at[i, 'release_date']), '%m/%d/%Y')

            columnName = 'day'
            if columnName in self.movies.columns:
                self.movies.at[i, columnName] = self.movies.at[i, 'release_date'].day
            else:
                nColumn = pd.DataFrame({columnName: [0] * self.movies.shape[0]})
                self.movies = pd.concat([nColumn, self.movies], axis=1)
                self.movies.at[i, columnName] = self.movies.at[i, 'release_date'].day

            columnName = 'month'
            if columnName in self.movies.columns:
                self.movies.at[i, columnName] = self.movies.at[i, 'release_date'].month
            else:
                nColumn = pd.DataFrame({columnName: [0] * self.movies.shape[0]})
                self.movies = pd.concat([nColumn, self.movies], axis=1)
                self.movies.at[i, columnName] = self.movies.at[i, 'release_date'].month

            columnName = 'year'
            if columnName in self.movies.columns:
                self.movies.at[i, columnName] = self.movies.at[i, 'release_date'].year
            else:
                nColumn = pd.DataFrame({columnName: [0] * self.movies.shape[0]})
                self.movies = pd.concat([nColumn, self.movies], axis=1)
                self.movies.at[i, columnName] = self.movies.at[i, 'release_date'].year'''
            try:
                datetime_object = datetime.strptime(self.movies.at[i, 'release_date'], '%m/%d/%Y')
                self.movies.at[i, 'release_date']= datetime_object.timetuple().tm_yday
            except:
                self.movies.at[i, 'release_date']= 0





            production_countries = json.loads(self.movies.iloc[i]['production_countries'])
            genres = json.loads(self.movies.iloc[i]['genres'])
            spoken_languages = json.loads(self.movies.iloc[i]['spoken_languages'])
            cast = json.loads(self.movies.iloc[i]['cast'])
            crew = json.loads(self.movies.iloc[i]['crew'])

            self.defineCategory(i, genres, 'name')
            self.defineCategory(i, production_countries, 'iso_3166_1')
            self.defineCategory(i, spoken_languages, 'iso_639_1')

            for c in cast:
                if(c['order'] >= 0 and c['order'] <= 2):
                    if(c['id'] in self.actorsDictionary):
                        self.actorsDictionary[c['id']] += 1
                    else:
                        self.actorsDictionary[c['id']] = 1

            for c in crew:
                if(c['job'] == 'Director'):
                    if(c['id'] in self.directorsDictionary):
                        self.directorsDictionary[c['id']] += 1
                    else:
                        self.directorsDictionary[c['id']] = 1



        self.actorsDictionary = sorted(self.actorsDictionary.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        self.actorsDictionary = dict(self.actorsDictionary[:200])

        self.directorsDictionary = sorted(self.directorsDictionary.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        self.directorsDictionary = dict(self.directorsDictionary[:50])


        for i in range(0, self.movies.shape[0]):
            cast = json.loads(self.movies.iloc[i]['cast'])
            crew = json.loads(self.movies.iloc[i]['crew'])

            for c in cast:
                if(c['id'] in self.actorsDictionary):
                    columnName = 'cast-' + str(c['id'])
                    if columnName in self.movies.columns:
                        self.movies.at[i, columnName] = 1
                    else:
                        nColumn = pd.DataFrame({columnName: [0] * self.movies.shape[0]})
                        self.movies = pd.concat([nColumn, self.movies], axis=1)
                        self.movies.at[i, columnName] = 1


            for c in crew:
                if(c['id'] in self.directorsDictionary):
                    columnName = 'crew-' + str(c['id'])
                    if columnName in self.movies.columns:
                        self.movies.at[i, columnName] = 1
                    else:
                        nColumn = pd.DataFrame({columnName: [0] * self.movies.shape[0]})
                        self.movies = pd.concat([nColumn, self.movies], axis=1)
                        self.movies.at[i, columnName] = 1



        to_drop = ['genres', 'production_countries', 'spoken_languages', 'cast', 'crew']
        self.movies.drop(to_drop, inplace=True, axis = 1)


    def dropUnnecessaryColumns(self, t):

        to_drop = ['homepage', 'id', 'original_title', 'overview', 'tagline', 'keywords', 'production_companies']
        if t=='t':
            self.movies.drop(to_drop, inplace=True, axis=1)
        else:
            self.test_movies.drop(to_drop, inplace=True, axis=1)


    def dropDuplicateColumns(self,t):
        to_drop = ['original_language', 'status', 'title_x', 'title_y', 'movie_id']
        if t=='t':
            self.movies.drop(to_drop, inplace=True, axis=1)
        else:
            self.test_movies.drop(to_drop, inplace=True, axis=1)

    def reformat(self,t):
        self.features=[]
        for i in self.movies.columns:
            self.features.append(i)

        self.features[-2], self.features[-4] = self.features[-4], self.features[-2]
        self.features[-1], self.features[-3] = self.features[-3], self.features[-1]
        if t=='t':
            self.movies = self.movies[self.features]
        else:
            self.test_movies = self.test_movies[self.features]

    def normalaize(self):
        for i in range(self.movies.shape[0]):
            for j in range(self.movies.shape[1]-6,self.movies.shape[1]-1):
                if (max(self.movies.iloc[:, j]) - min(self.movies.iloc[:, j])) != 0:
                    self.movies.iloc[i, j] = (self.movies.iloc[i, j] - min(self.movies.iloc[:, j])) / (
                                max(self.movies.iloc[:, j]) - min(self.movies.iloc[:, j]))



    def dropMissingRows(self):
        self.movies = self.movies[(self.movies['vote_count'] !=0).notnull() & (self.movies['runtime'] != 0).notnull()&(self.movies['vote_count'] !=0).notnull() ]
        if self.type=='c':
            self.movies = self.movies[self.movies['rate'].notnull()]
        else:
            self.movies = self.movies[self.movies['vote_average'].notnull()]



def handelMissingValues(movies):
    for key in movies.columns:
        mean = movies[key].describe()['mean']
        movies.loc[movies[key] == 0, key] = mean
        movies.loc[np.isnan(movies[key]), key] = mean
    return movies

def runDataCleaner(test_file1, test_file2,  type):
    if type=='c':
        d = DataCleaner("tmdb_5000_movies_classification.csv", "tmdb_5000_credits.csv", test_file1, test_file2, type)

        d.movies = pd.read_csv('trainC.csv')  # TrainC is data not numeric
        # loop on 2 files only (2 loops)

        for i in tqdm(['trainC_data.csv', 'testC_data.csv']):
            if (not os.path.exists(i)):  # If you have already created the dataset:
                d.movies = d.movies.iloc[:, 1:]  # beacuse coulmn 1 is the id
                d.defineCategories()
                d.normalaize()
                d.movies.to_csv(i)
            d.movies = pd.read_csv('testC.csv')
        # now d.movies has the train dataset in first loop, in the second loop d.moves has the test dataset
        train = handelMissingValues(pd.read_csv('trainC_data.csv').iloc[:, 1:])
        test = handelMissingValues(pd.read_csv('testC_data.csv').iloc[:, 1:])  # first coulmn is 0,1,2,3,...
    else:
        d = DataCleaner("tmdb_5000_movies_train.csv", "tmdb_5000_credits_train.csv", test_file1, test_file2, type)

        d.movies = pd.read_csv('trainR.csv')  # TrainC is data not numeric
        # loop on 2 files only (2 loops)

        for i in tqdm(['trainR_data.csv', 'testR_data.csv']):
            if (not os.path.exists(i)):  # If you have already created the dataset:
                d.movies = d.movies.iloc[:, 1:]  # beacuse coulmn 1 is the id
                d.defineCategories()
                d.normalaize()
                d.movies.to_csv(i)
            d.movies = pd.read_csv('testR.csv')
        # now d.movies has the train dataset in first loop, in the second loop d.moves has the test dataset
        train = handelMissingValues(pd.read_csv('trainR_data.csv').iloc[:, 1:])
        test = handelMissingValues(pd.read_csv('testR_data.csv').iloc[:, 1:])  # first coulmn is 0,1,2,3,...


    return train,test


def handle_test_and_train_missings(train, test,type):
    features = []
    for i in train.columns:
        features.append(i)
    # remove features that’s exist in testing and not exit in training # coulmns not exisit in training
    for i in test.columns:
        if i not in features:
            test.drop([i], inplace=True, axis=1)

    # Add dummy columns for features that’s not exist in testing but the model trained on it
    for i in features:
        if i not in test.columns:
            nColumn = pd.DataFrame({i: [0] * test.shape[
                0]})  # make coulmn with the same name in the train data, and make all rows in this coulmn=0
            test = pd.concat([nColumn, test], axis=1)


    features = features[:-1]
    X_train = train[features]
    X_test = test[features]
    if type=='c':
        y_train = train['rate']
        y_test = test['rate']
    else:
        y_train = train['vote_average']
        y_test = test['vote_average']

    return  X_train,X_test,y_train,y_test,features


