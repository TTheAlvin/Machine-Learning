# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:35:51 2022

@author: Student
"""

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD as PCA
Veg = pd.read_csv('0913_new2.csv')
# 把所有features列入考慮
x = Veg.iloc[:7752,9:33]
y = Veg.loc[:7751,'updown1']
y[:]=y[:].apply(lambda x: updown(x))
# 降維
from sklearn.preprocessing import StandardScaler,LabelEncoder
X_train, X_test, y_train, y_test = train_test_split(x_tsne, y, test_size=0.2)
x_pca = PCA(n_components=2).fit_transform(x)
tsne = manifold.TSNE(n_components=2,init='pca')
x_tsne = tsne.fit_transform(x)
# 畫圖
import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x_pca[:,0],x_pca[:,1])
sns.scatterplot(x_tsne[:,0],x_tsne[:,1],hue = y)
# SMOTE 平衡樣本
# conda install -c conda-forge imbalanced-learn
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(kind='regular',k_neighbors=2)
x_res, y_res = SMOTE(k_neighbors=1).fit_resample(X_train,y_train)

# 第二次畫圖比較
import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x_pca[:,0],x_pca[:,1])
sns.scatterplot(x_res[:,0],x_res[:,1],hue = y_res)
y_res[y_res==3].count()
