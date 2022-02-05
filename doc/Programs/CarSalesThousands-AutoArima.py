# -*- coding: utf-8 -*-
"""
Created on Thur Jan 21 02:50 2022

@author: Clive Smith
"""

import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import numpy as np


#Reads from xlxs file and puts in new dataframe
df = pd.read_excel("C:\\Users\\Csmit\\Documents\\MCAST\\Third Year\\Dissertation\\Dissertation\\dissertation-clive-smith\\doc\\Dataset\\CarSalesInThousandsAuto.xlsx")

#Puts data with category = 'Furniture' into new Dataframe
CarDF = df.loc[df['Vehicle_type'] == 'Car']

#Shows min and max of dates from rows
CarDF['Launch'].min(), CarDF['Launch'].max()

#Name unneeded columns from original df and removes them from furniture dataframe
cols = ['Manufacturer_model', 'Price', 'Horsepower', 'Fuel_efficiency', 'Vehicle_type']
CarDF.drop(cols, axis=1, inplace=True)

#Sorts values
CarDF = CarDF.sort_values('Launch')
#Checks null rows
CarDF.isnull().sum()

#Groups rows with same date and sums their sales figure up
CarDF = CarDF.groupby('Launch')['Sales'].sum().reset_index()

#Sets index as their date(timestamp)
CarDF = CarDF.set_index('Launch')
CarDF.index

#Takes the average sale figure of each month to make it as a MONTHLY time series and shows
y = CarDF['Sales'].resample('MS').mean()
y['2019':]

#Decompose to show trend, seasonality, residual and observed graphs
decomposition = seasonal_decompose(y, model = 'additive')
decomposition.plot()
pyplot.show()

#This library contains an auto_arima function that allows us to set a range of p,d,q,P,D,and Q values and then fit models for all the possible combinations. Then the model will keep the combination that reported back the best AIC value.
stepwise_model = auto_arima(y, start_p = 1, start_q = 1,
                            max_p = 3, max_q = 3, m = 12,
                            start_P = 0, seasonal = True,
                            d = 1, D = 1, trace = True,
                            error_action = 'ignore', 
                            supress_warnings = True, 
                            stepwise = True)

#The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a given set of data. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models.
#In this case the lowest AIC is 486.2063393665944
print(stepwise_model.aic())

#Split into training and testing datasets
train = y.loc['2010-02-01':'2019-10-01']
test = y.loc['2019-11-01':]

#Fits ARIMA model to training data
stepwise_model.fit(train)

#Predicts 2 periods + 36 new values (Test dataset number of rows)
test_forecast = stepwise_model.predict(n_periods=2)
future_forecast = stepwise_model.predict(n_periods=36)

#Concats predictions to test dataset rows into dataframe -- needs to be the size of the test dataset
test_forecast = pd.DataFrame(test_forecast,index = test.index,columns=['Prediction'])

#Frequency MS means month start
new = pd.DataFrame(future_forecast,index = pd.date_range(start='2020-01-01', end='2022-12-01', freq='MS'),columns=['Prediction'])

#Shows the mean squared error
mse = ((test_forecast['Prediction'] - test[1]) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
#Shows the RMSE - In this case - 420.08
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

pd.concat([y,test_forecast],axis=1).plot()

#Concats and plots
pd.concat([test,test_forecast],axis=1).plot()

#Concats and plots with training dataset (whole dataset)
pd.concat([y,new],axis=1).plot()

print(stepwise_model.summary())