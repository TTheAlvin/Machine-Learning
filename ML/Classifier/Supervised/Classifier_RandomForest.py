# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:50:45 2022

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
# def updown(y,ym):
#     if y <= ym:
#         return 0
#     else:
#         return 1
# y[:]=y[:].apply(lambda y: updown(y,ym))
# "設立漲跌lebel"
# +5%    ->   1
# +15%   ->   2
# -5%~5% ->   0
# -5%    -> (-1)
# -15%   -> (-2)
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

y[:]=y[:].apply(lambda x: updown(x))
# 開始normalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
model = RandomForestClassifier(max_depth=5, n_estimators=4)
model.fit(X_train, y_train)

X_test = sc.transform(X_test)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)

print('number of correct sample: {}'.format(num_correct_samples))
print('accuracy: {}'.format(accuracy))
print('con_matrix: \n{}'.format(con_matrix))


from sklearn.model_selection import learning_curve
# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(SVC(), 
                                                        X_train, 
                                                        y_train,
                                                        # Number of folds in cross-validation
                                                        cv=8,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()