# Simple Linear Regression
# Author = Michael Mourounas
# Task = Determine relationship between min and max temp,
#		 predict max temp from min temp

# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import data set
os.chdir('/users/bin/env')
data_set = pd.read_csv('Summary_of_Weather.csv')

ds_x_y = data_set.iloc[:, 5:7].values  # Isolate min and max temp variables

# Check and correct outliers
from scipy.stats import zscore

z_scores = np.abs(zscore(ds_x_y))
ds_x_y_o = ds_x_y[(z_scores < 3).all(axis=1)]  # Z score threshold = 3

# Split data
x = ds_x_y_o[:, 0]  # set x as min temp
y = ds_x_y_o[:, 1]  # set y as max temp

x = x.reshape(-1, 1)  # Reshape to avoid value error
y = y.reshape(-1, 1)

# Deal with missing data
# from sklearn.preprocessing import Imputer

# my_imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0)
# my_imputer = my_imputer.fit(x)
# x = my_imputer.transform(x)

# my_imputer = my_imputer.fit(y)
# y = my_imputer.transform(y)

# Build train and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# Feature scaling
# from sklearn.preprocessing import StandardScaler

# my_scaler_x = StandardScaler()
# x_train = my_scaler_x.fit_transform(x_train)
# x_test = my_scaler_x.transform(x_test)

# my_scaler_y = StandardScaler()
# y_train = my_scaler_y.fit_transform(y_train)
# y_test = my_scaler_y.transform(y_test)

# Determine relationship
# corr = np.corrcoef(x_train.T, y_train.T)

# Fit linear regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict test set results
y_pred = regressor.predict(x_test)

# Visualize the training set
plt.figure(1)
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Min vs Max Temp (C) (Training set)')
plt.xlabel('Min Temp (C)')
plt.ylabel('Max Temp (C)')

# Visualize the test set
plt.figure(2)
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, regressor.predict(x_test), color='blue')
plt.title('Min vs Max Temp (C) (Test set)')
plt.xlabel('Min Temp (C)')
plt.ylabel('Max Temp (C)')

plt.show()

print(regressor.coef_)
print(regressor.intercept_)
