# Course   : Data Science with Python
# Teacher  : Mr. Pouriya Baghdadi
# Student  : Mohammadreza Mashouf
# Session  : 11
# Question : 1 (HousingPricePrediction_Method 1)

'********************************************************************************************************'
# Algorithm : Linear Regression
# Featuer Selection Method : No feature has been removed

'********************************************************************************************************'
# Calling packages

import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

'********************************************************************************************************'
# Import data

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../S11_Assignment/housing.csv')
df = pd.read_csv(filename)
df = pd.DataFrame(pd.read_csv(filename))

'********************************************************************************************************'
# Preprocessing

# print('>>>>>>>>>> Dataset head: \n',df.head())
# print('>>>>>>>>>> Dataset shape: \n', df.shape)
# print('>>>>>>>>>> Null checking:\n', df.isnull().sum())

'********************************************************************************************************'
# Exploratory Data Analysis

# print('>>>>>>>>>> Dataset info: \n', df.info())
# print('>>>>>>>>>> Dataset describtion: \n', df.describe())
# Correlation check between features
corr = df.corr()
# print ('>>>>>>>>>> Data Correlation:\n',corr)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sn.axes_style('white'):
    fig, ax = plt.subplots(figsize=(14, 14))
    ax = sn.heatmap(corr, mask=mask, vmax=.3, square=True,annot=True, fmt='.0%')
    ax.set_title( 'Correlation Matrix')
plt.show()

'********************************************************************************************************'
# Feature Selection

# Non of the features have removed

'********************************************************************************************************'
# Normalization

scaler = StandardScaler()
# scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
# print(df.head())

'********************************************************************************************************'
# Linear Regression algorithm

# Model fitting
reg = LinearRegression()
x_train = df.iloc[:350, :-1]
y_train = df.iloc[:350, -1]
reg = reg.fit(x_train, y_train)

# Model Result
print('Train Score: ','{:.2f}'.format(reg.score(x_train, y_train)))
x_test = df.iloc[350:, :-1]
y_test = df.iloc[350:, -1]
y_train_pred = reg.predict(x_train)
y_test_pred = reg.predict(x_test)
print('Test Score: ','{:.2f}'.format(reg.score(x_test, y_test)))
from sklearn.metrics import r2_score
print('Linear Regression Test R2: ','{:.2f}'.format(r2_score(y_test,y_test_pred)))

# Train result visualization
plt.scatter (y_train,y_train_pred)
plt.xlabel('Actual Price')
plt.ylabel ('Predicted Price')
plt.title ('Train Result Scatter')
plt.grid()
plt.show()

# Test result visualization
plt.scatter (y_test,y_test_pred)
plt.xlabel('Actual Price')
plt.ylabel ('Predicted Price')
plt.title ('Test Result Scatter')
plt.grid()
plt.show()

# Cheking residuals
plt.scatter (y_test,y_test - y_test_pred, color ='r')
plt.xlabel('Actual Price')
plt.ylabel ('Residuals')
plt.title ('Actual Price VS. Residual')
plt.grid()
plt.show()

# Cheking residuals normality
residual = sn.histplot (y_test - y_test_pred, kde=True)
residual.set_title('Residuals Histogram')
residual.set(xlabel='Residuals', ylabel = 'Frequency')
plt.show()

