# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:16:30 2018

@author: Artem.Skachenko
"""


import pandas as pd

# Data
X = pd.read_csv("X_feat_sel.csv")
y = pd.read_csv("y.csv", header=None, names='y')


#########################################################
import  numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier


#########################################################
# tune tree
gbX = GradientBoostingClassifier()

tree_grid = {'max_depth':np.arange(1,12,1), 
             'min_samples_split':[2,10,25,50,100,150,250],
             'min_samples_leaf':[1,5,10,25,50,100,150,250],
             'max_leaf_nodes':[None, 5, 10, 20, 40, 80],
             'max_features':np.arange(2,10),
             'learning_rate':10 ** np.arange(-2,1, dtype=np.float),
             'n_estimators':[10,25,50,100,200]}
              
gsX = RandomizedSearchCV(gbX, tree_grid, cv=2, n_iter=200)

gsX.grid_scores_, gsX.best_params_, gsX.best_score_


#########################################################
# tune tree
gb0 = GradientBoostingClassifier(
             max_depth=4, 
             min_samples_split=150,
             min_samples_leaf=100,
             max_leaf_nodes=10,
             max_features=7,
             learning_rate=0.1)

tree_grid = {'n_estimators':[10,25,50,100,150,200,250,500]}
              
gs0 = GridSearchCV(gb0, tree_grid, cv=2)
gs0.fit(X, np.ravel(y))

gs0.grid_scores_, gs0.best_params_, gs0.best_score_







