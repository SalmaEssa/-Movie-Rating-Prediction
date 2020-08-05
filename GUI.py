from DataCleaner import *
from classifcation import*
from HandelMissingValues import*
credits_file="samples_tmdb_5000_credits_test.csv"
movies_reg="samples_tmdb_5000_movies_testing.xlsx"
movies_class="samples_tmdb_5000_movies_testing_classification.xlsx"
#for classification
train, test=runDataCleaner(movies_class,credits_file , 'c')
X_train,X_test,y_train,y_test,features=handle_test_and_train_missings(train, test,'c')
#classification_withoutPCA(X_train,X_test,y_train,y_test)
classification_with_PCA(X_train,X_test,y_train,y_test)


#for regression:

train, test=runDataCleaner(movies_reg,credits_file , 'r')
X_train,X_test,y_train,y_test,features=handle_test_and_train_missings(train, test,'r')
regression_without_pca(X_train,X_test,y_train,y_test,features)




