# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:12:57 2018

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
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


param_grid = [{'C': 10 ** np.arange(-3,4, dtype=np.float), 
               'penalty': ['l1', 'l2']}]
gs3 = GridSearchCV(logreg, param_grid, cv=2)

scores=pd.DataFrame()
for i in np.arange(1, X.shape[1]):
    X2 = SelectKBest(f_classif, k=i).fit_transform(X, np.ravel(y))
    gs3.fit(X2, np.ravel(y))
    score = pd.DataFrame({'n_feats':[i],'scr':[gs3.best_score_],
                          'C':[gs3.best_estimator_.C],
                          'penalty':[gs3.best_estimator_.penalty]})
    scores = pd.concat([scores, score], axis=0)
    del i,X2,score


gs3.best_params_





