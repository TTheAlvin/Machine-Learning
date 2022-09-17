# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:50:44 2022

@author: ALVIN
"""
import numpy as np
import pandas as pd
import math
Veg = pd.read_csv('0910_new3.csv')
# 把所有features列入考慮
x = Veg.iloc[:7752,9:33]
y = Veg.loc[:7751,'updown60']
# 以下用四個關鍵因素
X = Veg.loc[:7751,['temperature','humidity','windspeed','rainfall']]
# 後項式挑選11個Features(ANOVA)
X_b = Veg.loc[:7751,['pressure','volume','60temp','60hum','60wind','60rain']]
# 把價格分成大於整年平均'1'或小於等於平均'0'
# y = Veg.loc[:,"price"]
# ym = y.mean()
def updown(y,ym):
    if y <= ym:
        return 0
    else:
        return 1
y[:]=y[:].apply(lambda y: updown(y,ym))
# "設立漲跌lebel"
count    7752.000000
mean        0.048991
std         0.333961
min        -0.870482
25%        -0.163893
50%         0.006579
75%         0.209042
max         2.939759
Name: D+7, dtype: float64
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
y1 = Veg.loc[:7751,'D+1']
y.hist()
y.describe()
y[:]=y[:].apply(lambda x: updown(x))

# 將時間序列排整齊，由遠到近
# y7 = y7.sort_index(ascending=True,axis=0)
# 開始normalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
# Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
# X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)

print('number of correct sample: {}'.format(num_correct_samples))
print('accuracy: {}'.format(accuracy))
print('con_matrix:\n {}'.format(con_matrix))
print(y_pred[:5])