"""
Created on Sat Aug 20 20:11:50 2022

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
# 更改index資料的方法
y[:]=y[:].apply(lambda x: updown(x))
# 將資料按照index排序 ascending=True 從小到大
y = y.sort_index(ascending=True,axis=0)
y.describe()
y.hist()
print(Veg.info())
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

y[:]=y[:].apply(lambda x: updown(x))
# Veg1 = Veg1.assign(漲跌=y)
# 開始前我先觀察變數之間的關係-->1.變數距離 2.相似度(餘弦)
from sklearn.metrics.pairwise import euclidean_distances
Veg2 = X_b.values
Veg2 = Veg2.reshape(6,7752)
ecul_dists = euclidean_distances(Veg2)
Var = ['pressure','volume','60temp','60hum','60wind','60rain']
dist_df = pd.DataFrame(ecul_dists,columns=Var,index=Var)
def mod(vec):
    x = np.sum(vec**2)
    return x**0.5
def sim(vec1,vec2):
    s = np.dot(vec1,vec2) / mod(vec1) / mod(vec2)
    return s
cos_sim = sim(Veg2[0],Veg2[1])
# 計算樣本兩兩之間全部的相似度
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(Veg2)
sims_df = pd.DataFrame(cos_sim,columns=Var,index=Var)
# 關係視覺化
import seaborn as sns
sns.heatmap(sims_df,cmap="Blues",annot=True,fmt='.2f')
# 開始normalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
k_range = np.arange(start=1,stop=51,step=1)
weight_options = ['uniform', 'distance']
# param_grid = {'n_neighbors': k_range, 'weights': weight_options} 
param_grid = {'n_neighbors': 50, 'weights': 'distance'}
# 建立空模型
knn = KNeighborsClassifier(n_neighbors=5)  
grid = GridSearchCV(estimator = knn, param_grid = param_grid, cv=10, scoring='accuracy')
# corss_val_score 調參
scores = cross_val_score(grid, X_train,y_train, cv=5)
# 建立實模 
dir(grid)
grid.fit(X_train, y_train)
print('分數紀錄：',grid.cv_results_)
print('最佳引數:',grid.best_index_)
print('最佳分數:',grid.best_score_)  
print('最佳參數：',grid.best_params_)
print('最佳模型：',grid.best_estimator_)  
cv_res = pd.DataFrame(grid.cv_results_)
# 驗證檢果
cv_res.iloc[0, 7:17].mean()
cv_res.iloc[0, 7:17].std(ddof=1) 
# T-SNE
from sklearn.decomposition import TruncatedSVD as PCA
from sklearn import manifold
x_pca = PCA(n_components=2).fit_transform(Veg2)
tsne = manifold.TSNE(n_components=2,init='pca')
x_tsne = tsne.fit_transform(Veg2)


sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
model = KNeighborsClassifier(n_neighbors=50, weights='distance')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)
print('number of correct sample: {}'.format(num_correct_samples))
print('accuracy: {}'.format(accuracy))
print('confusion matrix: \n {}'.format(con_matrix))

