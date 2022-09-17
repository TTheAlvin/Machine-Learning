# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:12:04 2022

@author: ALVIN
"""
#### 5.1.1 多元線性迴歸 (Multiple Linear Regression by OLS)
import pandas as pd
import numpy as np

Veg = pd.read_csv('0910_new3.csv')
# 把所有features列入考慮
x = Veg.iloc[:7752,9:33]
y = Veg.loc[:7751,'updown1']
x = Veg.loc[:7751,['pressure', 'temperature', 'wind_direction', 'maximum_wind_direction', 'volume', '60temp', '60hum', '30hum']]
# Data normalization
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
X_train.columns.tolist()
index_Xtrain =  X_train.index.tolist()
index_Xtest =  X_test.index.tolist()
X_train.info()
X_train.describe()
# 數據標準化
from sklearn.preprocessing import MaxAbsScaler,StandardScaler,QuantileTransformer,RobustScaler
scaler = StandardScaler().fit(X_train,y_train)
# transform之後會把pd型態轉成Array 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# 在轉回pandas
X_train = pd.DataFrame(X_train,index=[index_Xtrain], columns=['pressure',
 'maximum_pressure',
 'minimum_pressure_x',
 'temperature',
 'maximum temperature',
 'minimum_temperature',
 'humidity',
 'minimum_humidity',
 'windspeed',
 'wind_direction',
 'maximum_windspeed_x',
 'maximum_wind_direction',
 'volume',
 'level',
 'alarms_count',
 'rainfall'])
X_test = pd.DataFrame(X_test,index=[index_Xtest],columns=['pressure',
 'maximum_pressure',
 'minimum_pressure_x',
 'temperature',
 'maximum temperature',
 'minimum_temperature',
 'humidity',
 'minimum_humidity',
 'windspeed',
 'wind_direction',
 'maximum_windspeed_x',
 'maximum_wind_direction',
 'volume',
 'level',
 'alarms_count',
 'rainfall'])
len(X_train) + len(X_train)
X_train.shape

#### 數據建模四部曲：Step 1. 載入類別函數, Step 2. 宣告空模, Step 3. 傳入數據集擬合/配適實模, Step 4. 以實模進行預測與應用

from sklearn.linear_model import LinearRegression # Step 1
lm = LinearRegression() # Step 2
# From the implementation point of view, this is just plain Ordinary Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares (scipy.optimize.nnls) wrapped as a predictor object.

pre = dir(lm) # 空模屬性及方法

lm.fit(X_train, y_train) # Step 3 (lm的內容在訓練樣本傳入配適後已發生變化！！！)

post = dir(lm) # 與pre的差異在於計算出來的物件

set(post) - set(pre) # 實模與空模的差異集合 {'_residues', 'coef_', 'intercept_', 'rank_', 'singular_' }

print(lm.coef_) # 24 slope parameters
print(lm.intercept_) # only one intercepts parameters
print(lm._residues) # 
print(lm.score(X_train, y_train)) # Step 4: Returns the coefficient of determination R^2 of the prediction. (訓練集的R2分數-->判定係數)

lmPred1 = lm.predict(X_train) # Step 4 預測測試樣本的價格準確度

from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# 模型績效
r_squared = r2_score(y_train, lmPred1) # train-data的判定係數 
rmse = sqrt(mean_squared_error(y_train, lmPred1))
# 印出模型績效(測試集的RMSE)
print('判定係數：{}'.format(r_squared))
print('均方根誤差：{}'.format(rmse))

#### 逐步迴歸 in Python
import stepwiseSelection as ss
# stepwiseSelection.py需放在當前工作路徑下，扮演套件/模組的角色，引入後簡稱為ss

#### 後向式迴歸(挑變量)
final_vars_b, iterations_logs_b = ss.backwardSelection(X_train, y_train) 
# 11 + intercept
# Final Variables: ['pressure', 'maximum_pressure', 'temperature', 'wind_direction', 'maximum_wind_direction', 'volume', '60temp', '60hum', '30temp', '30hum', 'rainfall']
#### 耗時模型，直接載入結果
import pickle # Python醃鹹菜套件
with open('final_vars_b.csv', 'wb') as f:
    pickle.dump(final_vars_b, f) # dump()表結果寫出去

with open('iterations_logs_b.csv', 'wb') as f:
    pickle.dump(iterations_logs_b, f) # dump()表結果寫出去

with open('final_vars_b.csv', 'rb') as f:
    final_vars_b = pickle.load(f) # load()表預存結果載入

#### 前向式迴歸(挑變量)
final_vars_f, iterations_logs_f = ss.forwardSelection(X_train,y_train) # 5 + intercept
# Final Variables :['intercept','rainfall','volume', 'wind_direction', '30temp', 'maximum_windspeed_x']
#### 耗時模型，直接載入結果
import pickle
with open('final_vars_f.csv', 'wb') as f:
    pickle.dump(final_vars_f, f)

with open('iterations_logs_f.csv', 'wb') as f:
    pickle.dump(iterations_logs_f, f)

with open('final_vars_f.csv', 'rb') as f:
    final_vars_f = pickle.load(f)

#### 逐步迴歸[降維]後的數據矩陣
X_train_b = X_train.loc[:, final_vars_b[1:]] # 1: 表不包括截距項
X_train_f = X_train.loc[:, final_vars_f[1:]] # 1: 表不包括截距項

#### 用statsmodels建立迴歸模型，其統計報表完整！
import statsmodels.api as sm # Step 1

lmFitAllPredictors = sm.OLS(y_train, X_train).fit() # Step 2 & 3

print(lmFitAllPredictors.summary()) # 看統計報表(sklearn用Ordinary Least Squares (scipy.linalg.lstsq)計算迴歸係數，但是無法提供統計檢定結果！)

#### 用前面後向式逐步迴歸挑出的10個變量擬合模型
reducedSolMdl = sm.OLS(y_train,X_train_b).fit()
print(reducedSolMdl.summary())
bwdSolMdl_sum = reducedSolMdl.summary()

#### 用前面前向式逐步迴歸挑出的9個變量擬合模型
fwdSolMdl = sm.OLS(y_train,X_train_f).fit()
print(fwdSolMdl.summary())
fwdSolMdl_sum = fwdSolMdl.summary()

# 檢視摘要報表的屬性與方法
[name for name in dir(bwdSolMdl_sum) if '__' not in name]

import re # re: regular expression package (Python強大的字串樣板正則表示式套件)
list(filter(lambda x: re.search(r'as', x), dir(bwdSolMdl_sum)))

#### 整個摘要報表轉為csv後存出
help(bwdSolMdl_sum.as_csv)

import pickle
with open('bwdSolMdl_sum.csv', 'wb') as f:
    pickle.dump(bwdSolMdl_sum.as_csv(), f)

# 把fwdSolMdl summary的各部分報表轉成html與DataFrame
bwdSolMdl_sum.tables # a list with three elements
len(bwdSolMdl_sum.tables) # 3
bwdSolMdl_sum.tables[0] # <class 'statsmodels.iolib.table.SimpleTable'>

#### 整體顯著性報表
bwdSolMdl_sum_as_html = bwdSolMdl_sum.tables[0].as_html()
bwdSolMdl_sum_as_html # str
pd.read_html(bwdSolMdl_sum_as_html, header=0, index_col=0)[0]

#### 模型係數顯著性報表
bwdSolMdl_sum_as_html_1 = bwdSolMdl_sum.tables[1].as_html()
pd.read_html(bwdSolMdl_sum_as_html_1, header=0, index_col=0)[0]

#### 殘差及其他統計值報表
bwdSolMdl_sum_as_html_2 = bwdSolMdl_sum.tables[2].as_html()
pd.read_html(bwdSolMdl_sum_as_html_2, header=0, index_col=0)[0]

#### ANOVA模型比較Ｆ檢定(https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html)
from statsmodels.stats.anova import anova_lm
anovaResults = anova_lm(fwdSolMdl, reducedSolMdl) # If None, will be estimated from the largest model. Default is None. Same as anova in R.
print(anovaResults) # 顯著(Pr(>F)很小)，故選擇後向式逐步迴歸模型

anovaResults = anova_lm(reducedSolMdl, lmFitAllPredictors)
print(anovaResults) # 不顯著(Pr(>F) = 1.0)，故選擇後向式逐步迴歸模型
