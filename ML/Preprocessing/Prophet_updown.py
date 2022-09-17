# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 09:23:59 2022

@author: Student
"""
import pandas as pd
from prophet import Prophet
dir(Prophet)
Veg = pd.read_csv('0915_new.csv')
Veg['date'] = Veg['date'].str.replace('/','-')

x = Veg.loc[:,['date','updown1']].values
x = pd.DataFrame(x,columns=['ds','y'])
# Prophet
m = Prophet()
m.fit(x)

# transform
future = m.make_future_dataframe(periods=1825)
future.tail()
md = dir(m)
# output
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# visualize all
fig1 = m.plot(forecast)

# visualize detail
fig2 = m.plot_components(forecast)





