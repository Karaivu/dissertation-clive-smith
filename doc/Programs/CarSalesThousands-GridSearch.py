# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 11:15 2022

@author: Clive Smith
"""

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib

print(np.__version__)
print(pd.__version__)
print(sm.__version__)
print(matplotlib.__version__)

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'black'

#Reads from xlxs file and puts in new dataframe
df = pd.read_excel("C:\\Users\\Csmit\\Documents\\MCAST\\Third Year\\Dissertation\\Dissertation\\dissertation-clive-smith\\doc\\Dataset\\CarSalesInThousandsGrid.xlsx")

#Shows min and max of dates from rows
df['DateObserved'].min(), df['DateObserved'].max()

#Sorts values by observation_df
df = df.sort_values('DateObserved')

#Checks null rows
df.isnull().sum()

#Groups rows with same date and sums their sales figure up
df = df.groupby('DateObserved')['SalesInThousand'].sum().reset_index()

#Sets index as their date(timestamp)
df = df.set_index('DateObserved')
df.index

#Takes the average sale figure of each month to make it as a MONTHLY time series and shows
y = df['SalesInThousand'].resample('MS').mean()
y['2017':]

#Visualise current data
y.plot(figsize=(15, 6))
plt.show()

#Decomposing time series into trend, seasonality and noise
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#Creates all possible combinations for SARIMAX in pdq combined with seasonal_pdq
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#Performs grid search to find the best possible parameters for the model
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
        #Best model results with params (1,1,1)x(0,0,1,12)12- AIC: 993.8
#Therefore we will fit this model as it shows that its the best one    
mod = sm.tsa.statespace.SARIMAX(y,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

#Plotting the diagnostics for any unusual behavior
results.plot_diagnostics(figsize=(16, 8))
plt.show()
#These plots show that residuals are near normally distributed 

#We will test the model forecasts by comparing real data to forecasted one from 2017 to end of dataset
#Afterwards we will plot to see the difference
pred = results.get_prediction(start=pd.to_datetime('2014-10-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

#This plots the confidence interval of the fitted parameters.
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Car Sales')
plt.legend()
plt.show()

#Shows the mean squared error - 1085.79
y_forecasted = pred.predicted_mean
y_truth = y['2014-10-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

#Shows the RMSE - Forecast vary by +- 32.95
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

#Forecast for the next 36 months or 3 years
pred_uc = results.get_forecast(steps=36)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='Observed', figsize=(14, 7), color = 'grey')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color = 'green')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Domestic Vehicle Sales')
plt.legend()
plt.show()