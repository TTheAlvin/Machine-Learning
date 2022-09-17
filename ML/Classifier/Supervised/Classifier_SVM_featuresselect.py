# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 09:40:01 2022

@author: Student
"""
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import math
Veg = pd.read_csv('0915_new.csv')
x = Veg.iloc[:7750,10:32]
y = Veg.loc[:7749,'updown1']
y.quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
y.describe()
y.mode()
y.median()
y.hist()
y.skew()
y = np.log1p(y)
yy = y.value_counts().sort_values(ascending=False)

df_log[‘price’] = np.log(df[‘price’])
sns.distplot(df_set['price'], fit=norm)
fig = plt.figure()

def updown(x):
    if  x >= 0.11 :
        x =1
    elif (x < 0.11) & (x > -0.11) :
        x = 0
    elif x < -0.11 : 
        x = -1  
    else:
        x = 2
    return  x
y[:]=y[:].apply(lambda x: updown(x))
### 1. Data Importing
Veg.info() # like str() in R
type(Veg) # pandas.core.frame.DataFrame
Veg.values
type(Veg.values) # numpy.ndarray
Veg.columns
type(Veg.columns) # pandas.core.indexes.base.Index
Veg.index
type(Veg.index) # pandas.core.indexes.range.RangeIndex
Veg.columns.values
type(Veg.columns.values) # numpy.ndarray
Veg[['pressure']]
type(Veg[['pressure']]) # pandas.core.frame.DataFrame (2D)
Veg['pressure']
type(Veg['pressure']) # pandas.core.series.Series (1D)

### 2. Data Sorting
Veg = Veg.drop(['Unnamed: 0'], axis=1)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
Veg.sort_index(axis=1).head()
# sort_values()
VegByIntl = Veg.sort_values(by=['pressure','temperature'], ascending=False)

#### 3. Summary Statistics
Veg.columns
Veg.median()
Veg.mean()
Veg.quantile([0, 0.25, 0.5, 0.75, 1])
x.skew()

y.quantile([0, 0.2, 0.4, 0.6, 0.8, 1])

Veg.pressure.mode() # mode-->其眾數
Veg.pressure.value_counts()

Veg.median()
Veg.mode()
# 類別變量分散程度 - 熵
from scipy.stats import entropy
entropy(x['pressure'].value_counts()) # 計算熵值時傳入次數分佈表

#### 3. Feature Transformation
# Standardization
from sklearn.preprocessing import StandardScaler,RobustScaler # Step 1
sc = StandardScaler() # Step 2
rs = RobustScaler() # Step 2
x = pd.DataFrame(rs.fit(x).transform(x),columns = x.columns)
sc.fit(Veg[['pressure', 'temperature']]) # Step 3
dir(rs)
rs.unit_variance
rs.scale_
pressure_std = sc.transform(Veg[['pressure', 'temperature']]) # Step 4

#### 4. Boxplot visualization by seaborn
Veg.info()
Veg.temperature.value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x=Veg['pressure'], y=Veg['temperature'])
plt.legend(loc='lower right')

#### Pie plot visualization by pandas
Veg.pressure.value_counts().plot.pie()

#### 5. Pick out low variance feature(s) 低變異(只對量化屬性)屬性過濾
# scikit-learn (from scipy下kits for machine learning) -> sklearn (sk stands for scikit) 低變異過濾會設定變異數值的門檻threshold
from sklearn.feature_selection import VarianceThreshold # Step 1

x.var()
sel=VarianceThreshold(threshold=0.16) # 0.16 -> 0.6, Step 2
print(sel.fit_transform(x)) # Steps 3 & 4
# help(sel)
# fit and transform on same object
sel.fit_transform(x).shape # (7752, 21), 3 low variance features 移除3個數值變數！(0.6會移除一個變量)
# Find the standard deviation and filter 直接計算各變數的標準差，再以邏輯值索引移除低變異數的變數
help(x.std)
x.std()
import numpy as np
~(x.std() < np.sqrt(0.16))
x_zvx = x.loc[:,~(x.std() < np.sqrt(0.16))]

#### 6. Transform skewed feature(s) by Box-Cox Transformation 偏斜屬性的BC轉換 (對量化變數計算偏態係數)
# 判斷變量分佈是否偏斜的多種方式：1. 比較平均數與中位數; 2. 最大值與最小值的倍數，倍比大代表數值跨越多個量綱/級order of mgnitude; 3. 計算偏態係數; 4. 繪製直方圖、密度曲線、盒鬚圖等變量分佈視覺化圖形; 5. 檢視分位數值quantiles, quartiles

x.skew(axis=0).sort_values()
x_skew = x.columns[x.skew(0) > 1] | x.columns[x.skew(0) < -1]
import math 
x_value = x.loc[:,['30rain','alarms_count','level','maximum_windspeed_x','rainfall','windspeed']].values
x_value = np.lo(x_value)
x_value = pd.DataFrame(x_value,columns=['30rain','alarms_count','level','maximum_windspeed_x','rainfall','windspeed'])
x_value.skew(0).sort_values()
 # 偏態係數高於1的屬性以及小於-1的屬性(array(['number_vmail_messages', 'total_intl_calls', 'number_customer_service_calls'], dtype=object))

#### 8. Box-Cox Transformation BC轉換
# 先試total_intl_calls前六筆(只接受一維陣列，自動估計lambda)
# λ=1等於沒做轉換，λ=0是對數轉換，λ=0.5， 是平方根轉換，λ=-1倒數轉換。
# 如果值等於0，Box-Cox會認為不適postive
要轉換的資料值必須是正數(postive)才能使用Box-Cox轉換。
from scipy import stats
print(x['rainfall'].head(10))
stats.boxcox(x['rainfall'].head(10))
x_finally = stats.boxcox(x['rainfall'])
# Output (array[transformed_values], lambda_used_for_BC_transform)
x_finally = pd.DataFrame(x_finally)
x_finally.median()
x_finally.mean()
x_finally.quantile([0, 0.25, 0.5, 0.75, 1])

x_finally.mode() # mode-->其眾數
x_finally.value_counts()

x_finally.median()
x_finally.mode()
