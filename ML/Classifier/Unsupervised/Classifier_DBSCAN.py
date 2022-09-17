# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:17:26 2022

@author: ALVIN
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
Veg = pd.read_csv('0910_new3.csv')
# 把所有features列入考慮
x = Veg.iloc[:7752,9:33]
y = Veg.loc[:7751,'updown1']
# Data normalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler,StandardScaler,QuantileTransformer,RobustScaler
scaler = StandardScaler().fit(x).transform(x)

sns.set()
neigh = NearestNeighbors(n_neighbors=5)
type(neigh) # sklearn.neighbors.unsupervised.NearestNeighbors
nbrs = neigh.fit(x)
type(nbrs) # sklearn.neighbors.unsupervised.NearestNeighbors same as above
[set(dir(nbrs)) - set(dir(neigh))] # An empty set
help(nbrs.kneighbors) # Finds the K-neighbors of a point. Returns indices of and distances to the neighbors of each point.
distances, indices = nbrs.kneighbors(x)
distances
indices
distances = np.sort(distances, axis=0)
distances = distances[:,1]


# DBSCAN
db = DBSCAN(eps=0.001, min_samples=5).fit(x)
m = db.fit(x)
labels = m.labels_
np.unique(clusters, return_counts=True)

# 雜訊樣本的群標籤為-1(numpy 產製次數分佈表的方式)
print(np.unique(labels, return_counts=True))
#  Note that -1 are noisy points
print('cluster on x {}'.format(labels))
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print('number of clusters: {}'.format(n_clusters))


# 把價格分成大於整年平均'1'或小於等於平均'0'
# y = Veg.loc[:,"price"]
# ym = y.mean()
# def updown(y,ym):
#     if y <= ym:
#         return 0
#     else:
#         return 1
# y[:]=y[:].apply(lambda y: updown(y,ym))
# 第一次               第二次
# +5%    ->   1       +20% -> 2
# +15%   ->   2       +20%~10% -> 1
# -5%~5% ->   0       -10%~+10% -> 0
# -5%    -> (-1)
# -15%   -> (-2)
# 第三次
# +50%      ->   2
# +0%~50%   ->   1
# -0%~-50%  ->   -1
# -50%      ->   -2
def updown(x):
    if  x >= 0.5 :
        x = 2
    elif (x >= 0) & (x<0.5) :
        x = 1
    elif (x < 0) & (x>-0.5) :
        x = -1
    elif (x < -0.5): 
        x = -2
    else:
        x = 3
    return  x
y7 = Veg.loc[:,'D+7']
y[:]=y[:].apply(lambda x: updown(x))