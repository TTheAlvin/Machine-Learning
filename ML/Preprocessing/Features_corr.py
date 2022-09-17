# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('0910_new3.csv')
y = data.loc[:7751,'updown1']
data = data.iloc[:7752,9:33]
data_f = data.loc[:,['pressure', 'wind_direction', 'volume', '60temp', '60hum']]

data = data.iloc[:,2:]
data.head(n=6)
y1 = data['D+1']
temp = data['temperature']
humi = data['humidity']
winds = data['windspeed']
rain = data['rainfall']
price = data['price']
temp.head(n=6)
# 數據normalization
from sklearn.preprocessing import StandardScaler,LabelEncoder
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
sc = StandardScaler()
data = sc.fit(data).transform(data)
data = pd.DataFrame(data)
# 產生變數時間序列與
# 調整漲跌幅
y1 = data['D+1']
y3 = data['D+3']
y5 = data['D+5']
y7 = data['D+7']
y1.index = pd.to_datetime(data['date'])
y3.index = pd.to_datetime(data['date'])
y5.index = pd.to_datetime(data['date'])
y7.index = pd.to_datetime(data['date'])
y1 = y1.sort_index(ascending=True,axis=0)
y3 = y3.sort_index(ascending=True,axis=0)
y5 = y5.sort_index(ascending=True,axis=0)
y7 = y7.sort_index(ascending=True,axis=0)
data.index = pd.to_datetime(data['date'])
data = data.sort_index(ascending=True,axis=0)
data.to_csv('30.csv',encoding='utf8')
y = pd.concat([y1,y3,y5,y7],axis=1)

y.to_csv('y.csv',encoding='utf8')  #存檔至New_Data.csv中
temp.index = pd.to_datetime(data['date'])
humi.index = pd.to_datetime(data['date'])
winds.index = pd.to_datetime(data['date'])
rain.index = pd.to_datetime(data['date'])
price.index = pd.to_datetime(data['date'])
temp.head(n=10)
# After ANOVA 變亮觀察
['pressure', 'maximum_pressure', 'temperature', 'wind_direction', 'maximum_wind_direction', 'volume', '60temp', '60hum', '30temp', '30hum', 'rainfall']]

x1 = data['temperature']
x2 = data['maximum_pressure']
x3 = data['pressure']
x4 = data['wind_direction']
x5 = data['maximum_wind_direction']
x6 = data['volume']
x7 = data['60temp'].dropna()
x8 = data['60hum'].dropna()
x9 = data['30temp'].dropna()
x10 = data['30hum'].dropna()
x11 = data['rainfall']

# 圖像化時間序列

temp.plot()
taiexPart = temp['2020-10-08':'2020-10-31']
taiexPart.head(5)
taiex2020 = temp['2008':'2010']
taiex2020.head()
taiex2020.describe()
taiex2020.hist()
humi.plot()
winds.plot()
rain.plot()
price.plot()

y.plot()
y.describe()
y.hist()
# 檢視變數分布
temp.describe()
humi.describe()
winds.describe()
rain.describe()
temp.hist()
humi.hist()
winds.hist()
rain.hist()
# 看整體分布
sns.heatmap(data_f, cmap='Reds', annot=True, linewidths=.5)
sns.heatmap(data.corr(),annot=True,fmt='.1f')
sns.pairplot(data_f)
sns.clustermap(data.corr(),annot=True,linewidths=.1,fmt='.1f')
g = sns.PairGrid(data_f)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
data.hist()
plt.tight_layout()
# 用熱力圖看特徵相關程度
plt.matshow(data.corr())
plt.xticks(range(data.shape[1]), data.columns, fontsize=8, rotation=0)
plt.yticks(range(data.shape[1]), data.columns, fontsize=8)
cb = plt.colorbar(shrink=.8)
cb.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.show()
# ADF test
from statsmodels.tsa.stattools import adfuller
import pandas as pd
df = pd.read_csv('0910_new3.csv',parse_dates=['date'])

dftem = df.loc[:,['temperature']]
result = adfuller(dftem ,autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
# 不穩定的話就會有單位根(unit root)   
if result[1] <= 0.05:  # 有顯著性，推翻虛無假設
     print("Data has no unit root and is stationary")
else:
     print("Data has a unit root and is non-stationary")
# ACF 藍色面積為95%信心水準下，如果柱狀點在面積之外代表很有影響95%以上影響  
# 觀察-'溫度'的自我相關性最高   
from statsmodels.graphics.tsaplots import *
plot_acf(x1,use_vlines=True,lags=30)  
plot_acf(x2,use_vlines=True,lags=30)   
plot_acf(x3,use_vlines=True,lags=30) 
plot_acf(x4,use_vlines=True,lags=30)   
plot_acf(x5,use_vlines=True,lags=30) 
plot_acf(x6,use_vlines=True,lags=30) 
plot_acf(x7,use_vlines=True,lags=30) 
plot_acf(x8,use_vlines=True,lags=30) 
plot_acf(x9,use_vlines=True,lags=30) 
plot_acf(x10,use_vlines=True,lags=30) 
plot_acf(x11,use_vlines=True,lags=30) 


y7 = y7.sort_index(ascending=True,axis=0)
y7.describe()
y7.hist()
