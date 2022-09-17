# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:18:31 2022

@author: Student
"""

import numpy as np
import pandas as pd
# 取var及label
Veg = pd.read_csv('新增漲跌.csv')
week ={}
Veg1 = Veg.iloc[:292,2:6]
x1 = Veg.iloc[:292,2:3]
x2 = Veg.iloc[:292,3:4]
x3 = Veg.iloc[:292,4:5]
x4 = Veg.iloc[:292,5:6]
y1 = Veg.loc[:291,"D+1"]
y2 = Veg.loc[:291,"D+3"]
y3 = Veg.loc[:291,"D+5"]
y4 = Veg.loc[:291,"D+7"]
# 觀察整體指標 
# temperature
x1_s = np.std(x1,ddof=1) # ddof=1母體標準差-1 , 2.85
x1_m = np.mean(x1) # 22.09
x1_v = np.var(x1) # 8.094 ,var = std**2
# humidity
x2_s = np.std(x2,ddof=1) # ddof=1母體標準差-1 , 3.52
x2_m = np.mean(x2) # 80.92
x2_v = np.var(x2) # 12.35
# windspeed
x3_s = np.std(x3,ddof=1) # ddof=1母體標準差-1 , 0.33
x3_m = np.mean(x3) # 1.75
x3_v = np.var(x3) # 0.11
# rainfall
x4_s = np.std(x4,ddof=1) # ddof=1母體標準差-1 , 11.65
x4_m = np.mean(x4) # 6.83
x4_v = np.var(x4) # 135.38
# 視覺化
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.35
fig = plt.subplots(figsize =(8, 6))

# set height of bar
s = [2.85, 3.52, 0.33, 11.65]
m = [22.09, 80.92, 1.75, 6.83]
v = [8.094, 12.35, 0.11, 135.38]

# Set position of bar on X axis
br1 = np.arange(len(s))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
print(br3)
# Make the plot
plt.bar(br1, s, color ='r', width = barWidth,
        edgecolor ='grey', label ='Standard')
plt.bar(br2, m, color ='g', width = barWidth,
        edgecolor ='grey', label ='Mean')
plt.bar(br3, v, color ='b', width = barWidth,
        edgecolor ='grey', label ='Variance')

# Adding Xticks
plt.xlabel('INDEX', fontweight ='bold', fontsize = 15)
plt.ylabel('SCORE', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(s))],
        ['temperture', 'humidity', 'windspeed','rainfall'])
 
plt.legend()
plt.show()

# D1~D7的比較

barWidth = 0.35
fig = plt.subplots(figsize =(8, 6))

# set height of bar
first = [1364,1.1]
second = [10794,57.09]
third = [899,0.69]
fourth = [3791,2.6]

# Set position of bar on X axis
br1 = np.arange(len(first))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
print(br4)

# Make the plot
plt.bar(br1, first, color ='r', width = barWidth,
        edgecolor ='grey', label ='D+1')
plt.bar(br2, second, color ='g', width = barWidth,
        edgecolor ='grey', label ='D+3')
plt.bar(br3, third, color ='b', width = barWidth,
        edgecolor ='grey', label ='D+5')
plt.bar(br4, fourth, color ='black', width = barWidth,
        edgecolor ='grey', label ='D+7')

# Adding Xticks
plt.xlabel('PREDICT AFTER DAYS', fontweight ='bold', fontsize = 15)
plt.ylabel('SCORE', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(first))],
        ['MSE', 'R-2'])
 
plt.legend()
plt.show()


