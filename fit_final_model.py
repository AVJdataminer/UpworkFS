# -*- coding: utf-8 -*-
"""Fit_final_model.py"""

import  pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import Imputer
import pickle
import warnings
warnings.filterwarnings("ignore")
import os

def final_model():
    #pre-process modeling data
    #train_filename =input("Enter training file name: ")
    train_filename ='client-trainingset-1561457457-219.csv'
    df = pd.read_csv(train_filename)
    print('preparing the data')
    dfo = df.select_dtypes(include=['object'])
    dfn = df.select_dtypes(exclude=['object'])
    imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    imputer = imputer.fit(dfn)
    dfna = pd.DataFrame(imputer.transform(dfn), columns = dfn.drop(['v262'], axis=1).columns)
    df_train = pd.concat([dfna, pd.get_dummies(dfo)], axis=1)
    df_train.shape
    #test_filename = input("Enter testing file name: ")
    test_filename = 'client-testset-1561457457-219.csv'
    dft = pd.read_csv(test_filename)
    dfto = dft.select_dtypes(include=['object'])
    dftn = dft.select_dtypes(exclude=['object'])
    imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    imputer = imputer.fit(dftn)
    dftna = pd.DataFrame(imputer.transform(dftn))
    df_test = pd.concat([dftna, pd.get_dummies(dfto)], axis=1)
    df_test.shape
    print('matching training and testing feature levels')
    Train2Test = df_train.columns.difference(df_test.columns)
    Test2Train = df_test.columns.difference(df_train.columns)
    print(Train2Test.shape)
    print(Test2Train.shape)
    addTest = pd.concat([df_train,pd.DataFrame(0,index =np.arange(len(df_test)), columns = Test2Train)],axis = 1)
    addTrain = pd.concat([df_test,pd.DataFrame(0,index =np.arange(len(df_train)), columns = Train2Test)],axis =1)
    print(addTrain.shape)
    print(addTest.shape)
    print('splitting training data into training and testing subsets')
    X=addTest.drop(['job_performance'], axis=1)
    y=addTest[['job_performance']].values  #convert to numpy arrays
    y=y.ravel()
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled=scaler.transform(X)
    # test and train split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=1)
    print('Fitting the model')
    """## Random Forest Regressor"""
    regressor = RandomForestRegressor(random_state=0, n_estimators=300, max_depth=None, max_features=10,
                                      min_samples_leaf=1, min_samples_split=2, bootstrap=False)
    regressor.fit(X_train, y_train)
    regressor.score(X_test, y_test)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    #print out the performance metrics
    print("Model MSE = " + str(mse))
    print('Predicting the testing dataset')
    test_scaled=scaler.transform(addTrain.drop(['job_performance'], axis=1))
    test_pred = regressor.predict(test_scaled)
    dft['job_performance'] = test_pred
    #writing out predictions
    dft.to_csv('predictions.csv')
    print('Completed writing predictions file')

if __name__ == '__main__':
    final_model()
