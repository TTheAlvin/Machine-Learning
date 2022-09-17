# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:56:27 2022

@author: ALVIN
"""
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from scipy import stats
import scipy.stats
Veg = pd.read_csv('0915_new.csv')
# 把所有features列入考慮
x = Veg.iloc[:7701,10:32]
y = Veg.loc[:7749,'updown1']
y = Veg.loc[:7700,'updown60']
y.quantile([0, 0.025, 0.475, 0.5, 0.975, 1])
y.describe()
y.hist()
kurtosis = stats.kurtosis(y) # 求峰度
skew = y.skew() # 求偏度
scipy.stats.shapiro(y)
print(skew,kurtosis)
y = np.log1p(y)
# 遞延1~60天的效果
Veg['updown']=Veg['price'].shift(-60)
Veg['updown60']=(Veg['updown']-Veg['price'])/Veg['price']
Veg.to_csv('0914_new2.csv',encoding='utf8')
# 以下用四個關鍵因素
x = Veg.loc[:7749,['temperature','humidity','windspeed','rainfall']]
# 後項式挑選11個Features(ANOVA)
x = Veg.loc[:7749,['pressure', 'maximum_pressure', 'temperature', 'wind_direction', 'maximum_wind_direction', 'volume', '60temp', '60hum', '30temp', '30hum','rainfall']]
# 自相關係數刪減--rainfall(10個)
x = Veg.loc[:7749,['pressure', 'maximum_pressure', 'temperature', 'wind_direction', 'maximum_wind_direction', 'volume', '60temp', '60hum', '30temp', '30hum']]
# 熱力圖刪減相關高的features(8個)
x = Veg.loc[:7749,['pressure', 'temperature', 'wind_direction', 'maximum_wind_direction', 'volume', '60temp', '60hum', '30hum']]
# 再做一次ANOVA(7個)
x = Veg.loc[:7749,['pressure', 'wind_direction', 'maximum_wind_direction', 'volume', '60temp', '60hum', '30hum']]
# xgb 特徵挑選後(6個)
x = Veg.loc[:7749,['pressure', 'wind_direction', 'maximum_wind_direction', 'volume', '60temp', '60hum']]
# 嘗試刪除windmax特徵挑選後(5個)-->有提升
x = Veg.loc[:7749,['pressure', 'wind_direction',  'volume', '60temp', '60hum']]
# 前項式挑選6個Features(ANOVA)
x = Veg.loc[:7749,['rainfall','volume', 'wind_direction', '30temp', 'maximum_wind_direction','temperature','60temp']]
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
# 第四次
# 0~25%  -> -2
# 25~50% -> -1
# 50~75% ->  1
# 75~100%->  2
# 第五次
# 0.0   -0.808660
# 0.2   -0.115607
# 0.4   -0.033902
# 0.6    0.028171
# 0.8    0.110348
# 1.0    0.962276
# 第六次
# 0.0   -0.554545
# 0.2   -0.109175
# 0.4   -0.033333
# 0.6    0.028571
# 0.8    0.116667
# 1.0    1.617647
# 第七次
# 設平均 0.0138 是持平
# 正負1 std 為漲跌 
# big_up = 0.188199 , 3 
# little_up = (x < 0.188) & (x >= 0.0138) , 2
# little_down = (x < 0.0138) & (x > -0.1605) ,1
# big_down = -0.160501 , 0 
第八次
# 設平均 0.0138 是持平
# up = 0.188199, 2
# hold = (x < 0.188199) & (x > -0.160501) ,1
# down = -0.160501, 0
y.mean()
y.median()
def updown(x):
    if  x >= 0.188199 :
        x = 3
    elif (x < 0.188199) & (x >= 0.0138) :  
        x = 2
    elif (x < 0.0138) & (x > -0.160501 ) :
        x = 1
    elif x <=  -0.160501  : 
        x = 0  
    return  x
def updown(x):
    if  x >= 0.188199 :
        x = 2
    elif (x < 0.188199) & (x > -0.160501) :
        x = 1
    elif x < -0.160501 : 
        x = 0  
    return  x
y[:]=y[:].apply(lambda x: updown(x))

# SVM
# we can change kernel to rbf, poly, linear
# C is margin-punishment,C 越大越不能超越界線
from sklearn.preprocessing import StandardScaler,RobustScaler
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
sc = StandardScaler()
rs = RobustScaler()
sc.fit(X_train)
rs.fit(X_train)
X_train = rs.transform(X_train)
X_test = rs.transform(X_test)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
# we can change kernel to rbf, poly, linear
# 減少 C 的值,會增加bias並提高variance(trade-off),因為越容易越界,分類越容易錯誤
# 變更徑向基底函數之參數gamma,分界的緊密程度
model = SVC(kernel='rbf', random_state=0, gamma=10, C=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)
print('number of correct sample: {}'.format(num_correct_samples))
print('accuracy: {}'.format(accuracy))
print('con_matrix: \n {}'.format(con_matrix))
print(y_pred[:5])
y_test = y_test.sort_index()
# 使用核pca,非線性的降維
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
scikit_kpca = KernelPCA(n_components=2,kernel='rbf',gamma=10)
x_sk = scikit_kpca.fit_transform(X_train,y_train)
plt.scatter(x_sk[y_train==0,0],x_sk[y_train==0,1],x_sk[y_train==0,2]color='blue',marker='o',alpha=0.5)
plt.scatter(x_sk[y_train==1,0],x_sk[y_train==1,1],color='red',marker='^',alpha=0.5)
plt.scatter(x_sk[y_train==1,0],x_sk[y_train==1,1],color='red',marker='^',alpha=0.5)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.tight_layout()
plt.show()

# Load libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
sns.heatmap(x,cmap='Reds',annot=True, fmt= '.2f')
# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(SVC(kernel='rbf', random_state=0, gamma=10, C=5), 
                                                        X_train, 
                                                        y_train,
                                                        # Number of folds in cross-validation
                                                        cv=10,
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
Train = plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
Test = plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"),plt.ylabel("Accuracy Score")
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
plt.legend(['Train','Test'])
plt.tight_layout()
plt.show()
