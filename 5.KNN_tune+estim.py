# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:00:33 2018

@author: Artem.Skachenko
"""


import pandas as pd
import os

os.chdir(r'C:\Users\artem.skachenko\Documents\_Never Backup\organize my job\learn\python\projects\test_project1')

# Data
X = pd.read_csv("X_feat_sel.csv")
y = pd.read_csv("y.csv", header=None, names='y')


#########################################################
import  numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()



param_grid = [{'n_neighbors': np.arange(25,30,5),
               'weights':['uniform', 'distance']}]
gs3 = GridSearchCV(knn, param_grid, cv=2)

scores=pd.DataFrame()
for i in np.arange(1, X.shape[1]+1):
    X2 = SelectKBest(f_classif, k=i).fit_transform(X, np.ravel(y))
    gs3.fit(X2, np.ravel(y))
    score = pd.DataFrame({'n_feats':[i],'scr':[gs3.best_score_],
                          'n_neighbors':[gs3.best_estimator_.n_neighbors],
                          'weights':[gs3.best_estimator_.weights]})
    scores = pd.concat([scores, score], axis=0)
    del X2,score,i



#########################################################
#########################################################
def self_cm(X,y):
    from sklearn.cross_validation import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    from sklearn.metrics import confusion_matrix as cm
    print(cm(y_test, gs3.predict(X_test)))

self_cm(X,y)



def self_roc(X,y):
    from sklearn.cross_validation import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve as roc
    fpr, tpr, thresholds = roc(y_test, gs3.predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr)
    plt.plot([0,1])
  
self_roc(X,y)










