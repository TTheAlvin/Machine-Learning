# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 21:02:41 2022
@author: ALVIN
"""
import numpy as np
import pandas as pd
#  數據標準化
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler,StandardScaler,QuantileTransformer,RobustScaler
#  建模
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
import xgboost as xgb
#  指標
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from math import sqrt
#  降維
from sklearn.decomposition import TruncatedSVD as PCA
from sklearn import manifold
#  畫圖
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# 取var及label
Veg = pd.read_csv('0901.csv')
y_price = Veg.loc[:7752,'price':'price']
# X = Veg.iloc[:7760,2:25]
# y = Veg.iloc[:7760,[26]]
# Price+1~7後面會有空值 所以x下面抓到7751
Veg2 = Veg.loc[:7752,['temperature','humidity','windspeed','rainfall']]
Veg3 = Veg.iloc[:7753,3:24]
y_price1 = Veg.loc[:7751,'price+1']
y_price3 = Veg.loc[:7751,'Price+3']
y_price5 = Veg.loc[:7751,'price+5']
y_price7 = Veg.loc[:7751,'Price+7']
y1 = Veg.loc[:7759,"D+1"]
y2 = Veg.loc[:7759,"D+3"]
y3 = Veg.loc[:7759,"D+5"]
y4 = Veg.loc[:7752,"D+7"]

# 把數據中%拿掉儲存
# y1 = Veg.iloc[:,11].values
# x = []
# for i in y1:
#     x.append(i.split('%')[0])
# print(x)
# x = np.array(x)
# Veg["D+7"] = x

# Veg.to_csv('新增漲跌.csv',encoding='utf8')  #存檔至New_Data.csv中
# 數據normalization
X_train, X_test, y_train, y_test = train_test_split(Veg3, y_price, test_size=0.2)
]]]# 使用多項式回歸
poly = PolynomialFeatures(degree=2)
poly.fit(X_train)
X_train = poly.transform(X_train)
poly.fit(X_test)
X_test = poly.transform(X_test)
# 漲幅有負值的時候使用 
# scaler = MaxAbsScaler().fit(X_train,y_train)
# 快速但相對不穩健的方法
# scaler = StandardScaler().fit(X_train,y_train)
# QuantileTransformer分開離群值的方法
# scaler = QuantileTransformer(random_state=0).fit(X_train,y_train)
# RobustScaler 預設25%-75%的是數據
scaler = RobustScaler().fit(X_train,y_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# 建立空模
model = LinearRegression()
model = linear_model.RidgeCV(10)
model = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
# model = make_pipeline(PolynomialFeatures(1),LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# The coefficients
# 模型的斜率及截距，分別儲存在model.coef_ 和 model.intercept_中。
print('Coefficients: {}\n'.format(model.coef_))
print('Intercept: {}\n'.format(model.intercept_))
print("MSE: {}".format((mean_squared_error(y_test, y_pred))))
print("RMSE: {}".format(sqrt(mean_squared_error(y_test, y_pred))))
print('R2 score: {}'.format(r2_score(y_test, y_pred))) 
print(model.score(X_test , y_test))
# 如果前面數據極度不准可以嘗試降低降維
x_pca = PCA(n_components=1).fit_transform(Veg2)
tsne = manifold.TSNE(n_components=3,init='pca')
x_tsne = tsne.fit_transform(Veg2)
# Adaboost
regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
regr.predict([[0,0,0,0]])
regr.score(X_train, y_train)
# XGboost
regressor = xgb.XGBRegressor(
    n_estimators=100,
    reg_lambda=1,
    gamma=0,
    max_depth=3
)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('R2: {}'.format(r2))
print('MSE: {}'.format(mse))
# 畫圖

# Plot outputs 三維等高線圖
x = Veg.loc[:,'price']
y = Veg.loc[:,'temperature']
def f(x, y):
    return np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
x, y=np.meshgrid(x,y)
z = f(x,y)
plt.contour(x,y,z, color='red')

plt.scatter(X_test, y_train,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
# Seaborn
sns.kdeplot(y1)

# 自行編譯
y_pred = line.predict(x_pca)
rss_p = sum((y_pred-y1))
line.fit(x_tsne,y1)
y_pred = line.predict(x_tsne)
rss_t = sum((y_pred-y1))

from sklearn.preprocessing import StandardScaler,LabelEncoder
line = LinearRegression(normalize=True)
line.fit(Veg1,y1)
y_pred = line.predict(Veg1)
rss = sum((y_pred-y1)**2)

from sklearn.linear_model import Ridge
ridge = Ridge(normalize=True)
ridge.fit(Veg1,y1)
y_pred1 = ridge.predict(Veg1)
rss1 = sum((y_pred-y1)**2)

