# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:35:30 2022

@author: ALVIN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import accuracy_score
Veg = pd.read_csv('0915_new.csv')
# 把所有features列入考慮
x = Veg.iloc[:7750,10:34]
y = Veg.loc[:7749,'updown1']

# Data normalization
from sklearn.preprocessing import MaxAbsScaler,StandardScaler,QuantileTransformer,RobustScaler
scaler = StandardScaler().fit(x).transform(x)
# k-means
kmeans = KMeans(n_clusters=5,n_init=10,max_iter=300,random_state=0)
kmeans = kmeans.fit(x)
labels = kmeans.predict(x)
centroids = kmeans.cluster_centers_
# 最佳 K
density  = []
for i in range(1,11):
    km = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    km.fit(x)
    density.append(km.inertia_)
plt.plot(range(1,11), density ,'o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia ')
plt.tight_layout()
plt.show()
print('centroids: {}'.format(centroids))
print('prediction on each data: {}'.format(labels))
print(np.unique(labels, return_counts=True))
accuracy = accuracy_score(y, labels)
num_correct_samples = accuracy_score(y, labels, normalize=False)
print('number of correct sample: {}'.format(num_correct_samples))
print('accuracy: {}'.format(accuracy))
# 畫圖
import matplotlib.pyplot as plt
plt.scatter(y[:, 0],y[:, 0],c=y, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5);

y.describe()
y.index = pd.to_datetime(Veg['date'])
# 把價格分成大於整年平均'1'或小於等於平均'0'
# # y = Veg.loc[:,"price"]
# # ym = y.mean()
# # def updown(y,ym):
# #     if y <= ym:
# #         return 0
# #     else:
# #         return 1
# # y[:]=y[:].apply(lambda y: updown(y,ym))
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
        x = 3
    elif (x >= 0) & (x<0.5) :
        x = 2
    elif (x < 0) & (x>-0.5) :
        x = 1
    elif (x < -0.5): 
        x = 0
    else:
        x = 4
    return  x
y[:]=y[:].apply(lambda x: updown(x))
def updown(x):
    if  x >= 0.116 :
        x = 2
    elif (x >= 0.028) & (x<0.116) :
        x = 1
    elif (x < 0.028) & (x>-0.033) :
        x = 0
    elif (x < -0.033) & (x>-0.109): 
        x = -1
    elif (x <= -0.109) : 
        x = -2   
    else:
        x = 3
    return  x
y[:]=y[:].apply(lambda x: updown(x))

