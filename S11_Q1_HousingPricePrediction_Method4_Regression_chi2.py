# Course   : Data Science with Python
# Teacher  : Mr. Pouriya Baghdadi
# Student  : Mohammadreza Mashouf
# Session  : 11
# Question : 1 (HousingPricePrediction_Method 3)

'********************************************************************************************************'
# Algorithm : Linear Regression
# Featuer Selection Method : Removing features with least chi2 score

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

# Chi2_Feature score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
x = df.iloc[:,:13] #features columns
y = df.iloc[:,-1]  #target column

'''If we want to identify the best features for the target variables.
We should make sure that the target variable should be int Values. 
That’s why we convert that into the integer.'''

y = np.round(df['MDEV'])
# Apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=9)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
# Concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Chi2_Score'] #naming the dataframe columns
# print('>>>>>>>>>> Feature score table - Sorted by score:\n',featureScores.sort_values('Chi2_Score',ignore_index=True,ascending=False)) #print best features
df.drop(columns=['NOX', 'RM', 'PTRATIO', 'CHAS'], inplace=True) # Remove features with least score
# print(df.head())

'********************************************************************************************************'
# Feature Importance

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
featureImportances = pd.DataFrame(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
featureImportances = pd.concat([dfcolumns,featureImportances],axis=1) # Concat two dataframes for better visualization
featureImportances.columns = ['Feature','Importances'] #naming the dataframe columns
# print('>>>>>>>>>> Feature Importance table:\n',featureImportances.nlargest(13,'Importances'))
# Plot graph of feature importances for better visualization
fig, ax = plt.subplots()
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh', ax = ax)
ax.set_title( 'Feature Importance Chart' )
plt.show()

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
# from sklearn.metrics import r2_score
# print('Linear Regression Test R2: ','{:.2f}'.format(r2_score(y_test,y_pred)))

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

