# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:50:44 2022

@author: ALVIN
"""
import numpy as np
import pandas as pd
Veg = pd.read_csv('0910_new3.csv')
# 把所有features列入考慮
x = Veg.iloc[:7752,9:33]
y = Veg.loc[:7751,'updown1']
# 把價格分成大於整年平均'1'或小於等於平均'0'
# y = Veg.loc[:,"price"]
# ym = y.mean()
def updown(y,ym):
    if y <= ym:
        return 0
    else:
        return 1
y[:]=y[:].apply(lambda y: updown(y,ym))
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

# 開始normalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
# Naviebayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# model = MultinomialNB()
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)

print('number of correct sample: {}'.format(num_correct_samples))
print('accuracy: {}'.format(accuracy))
print('con_matrix: {}'.format(con_matrix))
