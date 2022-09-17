# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:47:40 2022

@author: Student
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

Veg = pd.read_csv('0915_new.csv')
# 把所有features列入考慮
x = Veg.iloc[:7750,10:32]
# 嘗試刪除windmax特徵挑選後(5個)-->有提升
x = Veg.loc[:7749,['pressure','wind_direction','volume','60temp','60hum']]
y = Veg.loc[:7749,'updown1']
def updown(x):
    if  x >= 0.188199 :
        x = 2
    elif (x < 0.188199) & (x > -0.160501) :
        x = 1
    elif x < -0.160501 : 
        x = 0  
    return  x
y[:]=y[:].apply(lambda x: updown(x))
y.describe()
seed = 0
# prepare models
models = [] # tuple ('簡記名', 建模類別函數全名)
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf')))
models.append(('GMM', GaussianMixture()))
models.append(('XGB', XGBClassifier()))
models.append(('RANDF', RandomForestClassifier()))

# evaluate each model in turn
results = [] # 逐次添加各模型交叉驗證績效評量結果
names = []
scoring = 'balanced_accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True) # 十摺交叉驗證樣本切分
    # ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.
	cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
plt.xlabel('Methods')
plt.ylabel('Balanced_Accuracy')
ax.set_xticklabels(names)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
rs = RobustScaler()
rs.fit(X_train)
X_train = rs.transform(X_train)
X_test = rs.transform(X_test)
# we can change kernel to rbf, poly, linear
# 減少 C 的值,會增加bias並提高variance(trade-off),因為越容易越界,分類越容易錯誤
# 變更徑向基底函數之參數gamma,分界的緊密程度
model = SVC(kernel='rbf', random_state=0, gamma=10, C=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)
print('number of correct sample: {}'.format(num_correct_samples))
print('accuracy: {}'.format(accuracy))
print('con_matrix: \n {}'.format(con_matrix))

### 3.2.3_模型績效視覺化
# only can input 0 and 1 
import numpy as np
from sklearn.metrics import roc_curve, auc # area under curve

tpr, fpr, _ = roc_curve(y_pred, y_test)
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 45 degree straight line
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend(loc="lower right")
#fig.savefig('/tmp/roc.png')
plt.show()