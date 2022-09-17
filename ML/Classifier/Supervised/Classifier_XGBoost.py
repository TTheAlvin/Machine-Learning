# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 21:33:41 2022

@author: ALVIN
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
Veg = pd.read_csv('0910_new3.csv')
# 把所有features列入考慮
x = Veg.iloc[:7752,9:33]
y = Veg.loc[:7751,'updown7']
# 以下用四個關鍵因素
X = Veg.loc[:7751,['temperature','humidity','windspeed','rainfall']]
# 後項式挑選11個Features(ANOVA)
x = Veg.loc[:7751,['pressure','maximum_pressure','temperature','maximum temperature','minimum_temperature','humidity','windspeed','wind_direction','maximum_windspeed_x','maximum_wind_direction','volume','rainfall']]
# 特徵挑選後
x = Veg.loc[:7751,['pressure', 'wind_direction', 'maximum_wind_direction', 'volume', '60temp', '60hum']]
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
y7 = Veg.loc[:,'D+7']
y[:]=y[:].apply(lambda x: updown(x))

# 相關係數圖
plt.matshow(x.corr())
plt.xticks(range(x.shape[1]), x.columns, fontsize=10, rotation=90)
plt.yticks(range(x.shape[1]), x.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.show()

y[:]=y[:].apply(lambda x: updown(x))
y = y.sort_index(ascending=True,axis=0)
y.describe()
y.hist()
print(Veg.info())
# 摘要統計報表
report = x.describe(include='all')

# 載入建模套件與績效評估類別
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# 切分訓練集(80%) 與測試集(20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.2)
# xgboost 套件的資料結構DMatrix (後面計算量大交叉驗證使用)
data_dmatrix = xgb.DMatrix(data=x,label=y)
# 建立 XGBClassifier 模型
params = {"objective":"multi:softmax",'colsample_bytree': 0.5,
'learning_rate': 0.001, 'max_depth': 10, 'alpha': 10,
'silent': 1,'num_class':5} 
xgboostModel = XGBClassifier(n_estimators=150, params=params)
# 使用訓練資料訓練模型
xgboostModel.fit(X_train, y_train)
# 使用訓練資料預測分類
predicted = xgboostModel.predict(X_train)
# 預測成功的比例
print('訓練集: ',xgboostModel.score(X_train,y_train))
print('測試集: ',xgboostModel.score(X_test,y_test))
# xgboost.core.DMatrix
print(type(data_dmatrix))
                                                 
Objective candidate: survival:aft
Objective candidate: binary:hinge
Objective candidate: multi:softmax 讓XGBoost採用softmax目標函數處理多分類問題，同時需要設置參數num_class（類別個數）
Objective candidate: multi:softprob 和softmax一樣，但是輸出的是ndata * nclass的向量，可以將該向量reshape成ndata行nclass列的矩陣。沒行數據表示樣本所屬於每個類別的概率。
Objective candidate: rank:pairwise set XGBoost to do ranking task by minimizing the pairwise loss
Objective candidate: rank:ndcg
Objective candidate: rank:map
Objective candidate: survival:cox
Objective candidate: reg:gamma
Objective candidate: reg:squarederror
Objective candidate: reg:squaredlogerror
Objective candidate: reg:logistic 邏輯回歸
Objective candidate: binary:logistic 二分類的邏輯回歸問題，輸出為概率
Objective candidate: binary:logitraw 二分類的邏輯回歸問題，輸出的結果為wTx。
Objective candidate: reg:tweedie
Objective candidate: reg:linear 線性回歸。
Objective candidate: reg:pseudohubererror
Objective candidate: count:poisson 計數問題的poisson回歸，輸出結果為poisson分佈。在poisson回歸中，max_delta_step的缺省值為0.7。(used to safeguard optimization)
# 訓練參數同前
params = {"objective":"multi:softmax",'colsample_bytree': 0.5,
'learning_rate': 0.001, 'max_depth': 10, 'alpha': 10,
'silent': 1,'num_class':5} 
# 參數解釋
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 10,               # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 12,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
}
# k 摺交叉驗證訓練XGBoost
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10,
metrics="auc", as_pandas=True, seed=123, verbose_eval=False)

# 三次交叉驗證計算訓練集與測試集RMSE 的平均數和標準差
print(cv_results.head(15))
print(cv_results.tail(5))
print(cv_results["test-auc-mean"].tail(1))
# 訓練與預測回合數訂為10
xg_reg = xgb.train(params=params, dtrain=data_dmatrix,
num_boost_round=10)
# XGBoost 變數重要度繪圖
import matplotlib.pyplot as plt
ax = xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 10]
# plt.show()
fig = ax.get_figure()
# 分類準確度檢視表
def score(x, x_train, y_train, x_test, y_test, train=True):
    if train:
        pred=x.predict(X_train)
        print('Train Result:\n')
        print(f"Accuracy Score: {accuracy_score(y_train, pred)*100:.2f}%")
        print(f"Precision Score: {precision_score(y_train, pred)*100:.2f}%")
        print(f"Recall Score: {recall_score(y_train, pred)*100:.2f}%")
        print(f"F1 score: {f1_score(y_train, pred)*100:.2f}%")
        print(f"Confusion Matrix:\n {confusion_matrix(y_train, pred)}")
    elif train == False:
        pred=x.predict(X_test)
        print('Test Result:\n')
        print(f"Accuracy Score: {accuracy_score(y_test, pred)*100:.2f}%")
        print(f"Precision Score: {precision_score(y_train, pred)*100:.2f}%")
        print(f"Recall Score: {recall_score(y_test, pred)*100:.2f}%")
        print(f"F1 score: {f1_score(y_test, pred)*100:.2f}%")
        print(f"Confusion Matrix:\n {confusion_matrix(y_test, pred)}")
from xgboost import XGBClassifier
xg1 = XGBClassifier()
xg1=xg1.fit(X_train, y_train)
score(xg1, X_train, y_train, X_test, y_test, train=False)
