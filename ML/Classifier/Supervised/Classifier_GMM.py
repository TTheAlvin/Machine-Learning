# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:49:22 2022

@author: Student
"""

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import mixture
import pandas as pd
import numpy as np

Veg = pd.read_csv('0913_new.csv')
x = Veg.iloc[:7752,10:34]
y = Veg.loc[:7751,'updown1']
def updown(x):
    if  x >= 0.50 :
        x = 4
    elif (x >= 0.028) & (x<0.50) :
        x = 3
    elif (x < 0.028) & (x>-0.033) :
        x = 2
    elif (x < -0.033) & (x>-0.50): 
        x = 1
    elif (x <= -0.50) : 
        x = 0   
    else:
        x = 3
    return  x
y[:]=y[:].apply(lambda x: updown(x))

gmm = mixture.GaussianMixture(n_components=5).fit(x)
X_pred = gmm.predict(x)

accuracy = accuracy_score(y, X_pred)
print('accuracy: {}'.format(accuracy))
print(X_pred)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

X_train = np.load('data.npy')

print('X_train shape: {}'.format(X_train.shape))
print(X_train[:5,:])
print('\n\n')

gmm = GaussianMixture(n_components=2)
gmm.fit(X_train)

print('means:\n{}'.format(gmm.means_))
print('covariances:\n{}'.format(gmm.covariances_))

X, Y = np.meshgrid(np.linspace(-1, 6), np.linspace(-1,6))
XX = np.array([X.ravel(), Y.ravel()]).T
Z = gmm.score_samples(XX)
Z = Z.reshape((50,50))

plt.contour(X, Y, Z)
plt.scatter(X_train[:, 0], X_train[:, 1])

plt.show()