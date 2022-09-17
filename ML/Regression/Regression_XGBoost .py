# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 21:33:41 2022

@author: ALVIN
"""
import numpy as np
import pandas as pd

Veg = pd.read_csv('0910_new3.csv')
# 把所有features列入考慮
x = Veg.iloc[:7752,9:33]
y = Veg.loc[:7751,'updown1']
# 以下用四個關鍵因素
X = Veg.loc[:7751,['temperature','humidity','windspeed','rainfall']]
# 後項式挑選11個Features(ANOVA)
X_b = Veg.loc[:7751,['pressure','maximum_pressure','temperature','maximum temperature','minimum_temperature','humidity','windspeed','wind_direction','maximum_windspeed_x','maximum_wind_direction','volume','rainfall']]
# 更改index資料的方法
y[:]=y[:].apply(lambda x: updown(x))
# 將資料按照index排序 ascending=True 從小到大
y = y.sort_index(ascending=True,axis=0)
y.describe()
y.hist()
print(Veg.info())
# 摘要統計報表
report = x.describe(include='all')

# 載入建模套件與績效評估類別
import xgboost as xgb 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
# xgboost 套件的資料結構DMatrix (後面計算量大交叉驗證使用)
data_dmatrix = xgb.DMatrix(data=x,label=y)

# xgboost.core.DMatrix
print(type(data_dmatrix))

# 切分訓練集(80%) 與測試集(20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
                                                    
# 宣告xgboost 迴歸模型規格 
# boosting max_depth, how about bagging?
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',
colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5,
alpha = 10, n_estimators = 10) 
# XGBoost objective methods
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
# 傳入資料擬合模型
xg_reg.fit(X_train,y_train)
dir(xg_reg)
dt = pd.DataFrame(xg_reg.feature_importances_.reshape(1, -1), columns=x.columns)

# 預測測試集資料
preds = xg_reg.predict(X_test)
# 傳入實際值與預測值向量計算均方根誤差
mse = np.array(mean_squared_error(y_test, preds))
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print("RMSE = %f" % (rmse))
print("MSE = %f" % (mse))
print('R2 score: {}'.format(r2_score(y_test ,preds)))
# 訓練參數同前
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,
'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10,# "eval_metric":"auc",
'silent': 1} 
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
# searchCV
xgb_params={'max_depth':[4,5,6,7],
           'learning_rate':np.linspace(0.03,0.3,10),
           'n_estimators':[100,200]}
xgb_search = GridSearchCV(xg_reg,
                          param_grid= params,
                          scoring='rmse',
                          cv=5)
xgb_search.fit(X_train,y_train)
# k 摺交叉驗證訓練XGBoost
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10,metrics="rmse",
as_pandas=True, seed=123, verbose_eval=False)
print(cv_results)
dir(cv_results)

# 三次交叉驗證計算訓練集與測試集RMSE 的平均數和標準差
print(cv_results.head(15))
print(cv_results.tail(5))
print(cv_results["test-rmse-mean"].tail(1))

# 訓練與預測回合數訂為10
xg_reg = xgb.train(params=params, dtrain=data_dmatrix,
num_boost_round=10)
# XGBoost 變數重要度繪圖
import matplotlib.pyplot as plt
ax = xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5,5]
# plt.show()
fig = ax.get_figure()






