# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:28:54 2022

@author: ALVIN
"""

先備知識

Rows為行(上到下)=0
Columns為列(左到右)=1
讀檔
df = pd.read_csv('./csv檔案位置')                  #可讀CSV和txt檔案
df = pd.read_csv('./csv檔案位置',header=None)      #如果想要自行設定標題列(最左列)，應該先將header設定為None，再自行指定Column為header
df = pd.read_excel('./xlsx檔案位置')  
df = pd.read_html('./html檔案位置')    #html中的表格
df = pd.read_json('./json檔案位置')
df = pd.read_sql('./sql檔案位置')
df = pd.read_clipboard('網址或剪貼簿')  #讀取網頁表格或剪貼簿的內容
初始化

import pandas as pd #引入Pandas模組 as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", 1000)    #設定最大能顯示1000rows
pd.set_option("display.max_columns", 1000) #設定最大能顯示1000columns
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
# 指定默認字形：解決plot不能顯示中文問題
mpl.rcParams['axes.unicode_minus'] = False
讀取檔案

以OpenData的AQI資料為例:
至https://opendata.epa.gov.tw/ws/Data/AQI/?$format=csv 下載回CSV檔
df=pd.read_csv(r'C:\Users\Yanwei\Desktop\AQI.csv')
df=pd.read_csv(‘C:/Users/Yanwei/Desktop/AQI.csv’) #讀取AQI.csv
兩個df的差別
1)在前方加r，就不用反斜線
2)使用反斜線就不用加r
資料操作

基本操作
df.head(10)  #顯示出前10筆資料，預設值為5筆資料
df.tail(10)  #顯示倒數10筆資料
df.shape()   #顯示出資料共有(X行,Y列)
len(df)      #顯示資料總筆數
df.dtypes    #顯示資料類型
df.select_dtypes("string") #選取字串類型的資料(新功能)
df.describe()#顯示統計數字(最大、最小、平均......等)
df[['AQI']]  #顯示Columns(列)為AQI的數據
df.AQI       #顯示Columns(列)為AQI的數據
df.rename(columns={'舊欄位名稱': '新欄位名稱'}) #修改欄位名稱
df.columns                        #顯示有哪些欄位
df.columns = ['XXX','XXX', 'XXX'] #新增欄位
df.T         #行與列互換，等同於df.transpose()
             #例如df.describe().transpose()
df.info()    #顯示資料的狀態與資訊
df.info(memory_usage='deep') #顯示記憶體使用狀況
df.query('A < 0.5 and B < 0.5')      #查詢A<0.5且B<0.5的資料
df.corr()['PM25'].sort_values()      #顯示PM2.5與其他欄位間的相關係數
df.get_dummies      #One-hot編碼
df.AQI.values       #將資料轉成numpy的array
df.Danger.unique()  #找出唯一值
df.duplicated()     #顯示重複的資料
df.drop_duplicates()#刪除重複的資料
df.drop_duplicates(['Name']) #刪除Name欄位重複的資料
df.value_counts()  #查看有哪些不同的值，並計算每個值有多少個重複值
#檢查AQI欄位當中，包含NaN資料的比例
df["AQI"].value_counts(dropna=False, normalize=True)
groupby方法
dfTotal=df.groupby(['AQI','PM25']).sum() 
#合併種類的名稱，並且顯示該名稱欄位的所有數量總合
dfTotal.sum()                        
#加總所有欄位成一數字
-------------------------------------------------------------------
df_Danger_PM25=df[df.PM25>35.5].groupby("Danger_Air")
#合併所有PM2.5數值>35.5以上的資料成一個新欄位「Danger_Air」df_Danger_PM25["AQI"].sum()
#查詢Danger_Air中，所有的AQI值總合
iloc,loc,ix方法
df.iloc[4]   #顯示第4筆資料的所有數據 
df1 = df.set_index(['測站'])       #將測站設定為索引(即擺到第一行第一列)
df1 = df1.reset_index(['測站'])    #恢復原本設置
df1.loc['左營']                    #列出所有左營的數據
df.loc[df['name'] == 'Jason']     #列出Name為Jason的資料
找極端的排序
(例如:前n個大的值或n個最小的值，實際一點的例子像是查詢班上的前三名是誰)
df.nlargest(3,'HEIGHT')    #查詢HEIGHT欄位中數值前3大的
df.nsmallest(3,'WEIGHT')   #查詢WEIGHT欄位中數值前3小的
刪除資料
df.drop(labels=['SO2','CO'],axis='columns') #刪除SO2和CO這兩個欄位
df.drop(labels=['SO2','CO'],axis='columns',inplace=True)
df=df.drop_duplicates()                     #刪除重複的資料
df.drop(df.index[-1])                       #刪除最後一筆資料
axis=0和asxis='row'一樣
axis=1和axis='columns'一樣
使用inplace=True才會把原本的資料改變
處理Nan資料
df.isnull()                          #判斷是否有遺失值
df.hasnans                           #檢查是否有NaN資料
df.isnull().any()                    #迅速查看是否有遺失值(大筆數資料)
df.isnull().sum()                    #查看每個欄位有幾個缺失值
df.dropna()                          #刪除NaN的資料
df=df.dropna()                       #將刪除後的資料存到變數
df.dropna(axis=1)                    #删除所有包含空值的列
df.dropna(axis=0)                    #删除所有包含空值的行
df.dropna(how='all')                 #只刪除全部欄位都是NaN的列
df.dropna(thresh=4)                  #刪除小於4項缺失值的行
df.dropna(subset=['PM25'])           #只刪除PM25欄位中的缺失值df=df.fillna(0)                      #把NaN資料替換成0 
df=df.fillna(method='pad')           #填入前一筆資料的數值
df=df.fillna(method='bfill')         #填入下一筆資料的數值
df['PM25']=df['PM25'].fillna((df['PM25'].mode())) #填入眾數
df['PM25'] = df['PM25'].interpolate()#使用插值法填入數字(用函數方式)
df['PM25'].fillna(value=df['PM25'].mean()) #把NaN值改成該屬性的所有平均值
Sort排序
df.sort_index(ascending=True).head(100)         #升階排序
df.sort_index(ascending=False).head(100)        #降階排序
#指定欄位進行由小到大的排序
dfSort=df.sort_values(by='物種中文名',ascending=False).head(100) 
#指定多個欄位進行由小到大的排序
dfSort=df.sort_values(by=['名稱1', '名稱2'], ascending=False)
備註
基本上df[['AQI']]和df.AQI功能一樣

loc:以行列標題選擇資料(隻對字串類型有用)
ix擁有iloc與loc的功能
iloc:以第幾筆來選擇資料(隻對數值類型有用)
使用函數

import numpy as np
df.apply(np.sqrt)             #計算平方根
df.apply(np.sum)              #計算總合
df['A'].apply(np.sqrt)        #計算A欄位的平方根
處理時間序列

df["date"] = pd.to_datetime(df["date"]) #轉換字元為日期時間型別df.sort_values("date")                  #用時間先後序列排序資料
df = df.set_index("date", drop=True)    #將日期時間擺放至列索引值
進行分類切割

op_labels = ['shyttte', 'moderate', 'good']
category = [0.,4.,7.,10.]
movies_df['imdb_labels'] = pd.cut(movies_df['imdb_score'], labels=op_labels, bins=category, include_lowest=False)
print(movies_df[['movie_title', 'imdb_score', 'imdb_labels']])[209:220]
>>>
movie_title     imdb_score     imdb_labels
209     Rio 2               6.4            moderate
210     X-Men 2             7.5             good
211     Fast Five           7.3             good
212     Sherlock Holmes:..  7.5             good
213     Clash of the...     5.8            moderate
214     Total Recall        7.5             good
215     The 13th Warrior    6.6            moderate
216     The Bourne Legacy   6.7            moderate
217     Batman & Robin      3.7             shyttte
218     How the Grinch..    6.0            moderate
219     The Day After T..   6.4            moderate
資料型態轉換

如果屬性是Object，如何改成數值屬性
df["屬於Object的欄位"] = pd.to_numeric(df.屬於Object的欄位, errors='coerce') #即可將該物件轉成數值屬性
例如
df["AQI"] = pd.to_numeric(df.AQI, errors='coerce')
轉換屬性成字串或數字
df["AQI"] = df["AQI"].astype(str)      #數值變成字串
df["AQI"] = df["AQI"].astype(int)      #字串變成數值
df['Date']= pd.to_datetime(df['Date']) #轉換成時間
字串處理

大小寫與字串變更
df['PM2.5'].str.title() #讓字串第一個字為大寫
df['PM2.5'].str.lower() #讓字串全部變成小寫
df['PM2.5'].str.upper() #讓字串全部變成大寫
df['PM2.5']=df['PM2.5'].str.replace("原有字串","欲改變成的字串")
找出資料
df[df.AQI.startswith('高雄市')]  #顯示出高雄市開頭的資料df[df.AQI.endswith('高雄市')]    #顯示出高雄市做為結尾的資料
來一點複雜操作

#20200503更新：建議用df.query('AQI<50 and AQI>60')查詢，程式碼會比較乾淨
df[['AQI','WindSpeed']]#顯示Columns(列)為AQI及WindSpeed的數據df[df.AQI<50]          #顯示AQI<50的數值
df[(df.AQI < 30)&(df.WindSpeed>2)] #列出AQI值大於30且風速大於2的數值
df['AQI'] / 2 #將所有AQI值除以2(+,-,*,/皆適用)
#從所有AQI介於10到30的資料之中，抽出5筆資料
df[df["AQI"].between(10, 30, inclusive="neither")].sample(5)
-----------------------------------------
AQI_filter = df['AQI']>60 #使用布林，當AQI>60為True，<60為False

Bad_AQI= df[AQI_filter] #將過濾後的數值存入至Bad_AQI
Bad_AQI.head() #隻顯示AQI>60的資料
AQI_filter_2 = (df['AQI']>60)&(df['PM2.5']>40) 
#使用布林，條件是AQI>60且PM2.5數值超過40
Bad_AQI_PM= df[AQI_filter_2] #將過濾後的數值存入至Bad_AQI_PM
Bad_AQI_PM.head() #隻顯示AQI>60且PM2.5>40的資料
針對資料進行Encoding

Encode the object as an enumerated type or categorical variable.

codes, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b'])
>>> codes
array([0, 0, 1, 2, 0]...)
>>> uniques
array(['b', 'a', 'c'], dtype=object)
Mask遮罩

# treat ages outside the 50–60 range (there are two, 49, and 66) as data entry mistakes and replace them with NaNs.
# So, mask replaces values that don't meet cond with other.
# 先選出年紀介於50~60的選出來，搭配"~"符號選出50~60之外的，將其值變成NaN
>>> ages.mask(cond=~ages["ages"].between(50, 60), other=np.nan)
搜尋資料

#搜尋location欄位中，包含"高雄"或"台南"的資料，並且將其存成變數loc
str_choice = "高雄 | 台南" 
loc=df[df['location'].str.contains(str_choice, na=False)]
合併資料框

Location = ["高雄", "台南", "台中", "台北"]
PM25 = [30, 35, 20, 15]
NOx= [1, 2, 3, 2, 1]
PM25_df = pd.DataFrame()
PM25_df["Location"] = Location
PM25_df["PM25"] = PM25
NOx_df = pd.DataFrame()
NOx_df["Location"] = Location
NOx_df["NOx"] = NOx
pd.merge(PM25_df, NOx_df, on='Location')
#pd.merge()：將兩個資料框合而為一
合併資料集

Jan_sales = pd.read_csv("jan_sales.csv")
Feb_sales = pd.read_csv("feb_sales.csv")
#合併有共同欄位的資料集(例如合併一月與二月的銷售資料)
pd.concat([Jan_sales, Feb_sales])   
#合併有共同index的欄位
table_1.join(table_2, on='customer_id', how='left')
進度條(顯示pandas的操作狀態)

from tqdm import tqdm 
tqdm.pandas() 
df = pd.read_csv('pathTOfile') 
df.progress_apply(lambda x: x, axis=1)
特殊表格

樞紐關係表
df.pivot_table
交叉表
df.crosstab
簡單繪圖與存檔

資料視覺化
#進行繪圖(X軸為地點,Y軸為AQI數值)
df.plot(x='SiteName', y=['AQI'])
#製作散布圖,X軸風速,Y軸為PM2.5指數
df.plot(kind = 'scatter', x = 'WindSpeed', y = 'PM2.5', title = '風速與PM2.5之關係')
#繪製PM2.5,PM10,AQI於同一張圖上
df = df[['PM25', 'PM10', 'AQI']]
df.plot()
#繪製長條圖
df.plot(kind='bar')   #垂直
df.plot(kind='barh')  #水平
存檔
df.to_csv('New_Data.csv',encoding='utf8')  #存檔至New_Data.csv中
df.to_json('New_Data.json', encoding='utf8')#存檔至New_Data.json
df.to_excel('New_Data.xlsx', encoding='utf8')#存檔至New_Data.xlsx
df.to_html('New_Data.html', encoding='utf8') #存檔至New_Data.html
df.to_markdown('New_Data.md')            #存檔至New_Data.md(新功能)
                                         #必須安裝pip install tabulate
con = sqlite3.connect('mydatabase.db')    #存檔至mydatabase.db
df.to_sql('users', con)
解決儲存的中文csv檔用Excel打開是亂碼
df.to_csv('New_Data.csv', encoding='utf_8_sig')(或是cp950)