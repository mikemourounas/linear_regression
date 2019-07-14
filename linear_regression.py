# Simple Linear Regression
# Find relationship between temp C and salinity; predict temp C

# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import data set
os.chdir('/users/michaelmourounas/Desktop/')
data_set = pd.read_csv('bottle.csv')

x = data_set.iloc[:, 6].values  # set x as salinity
y = data_set.iloc[:, 5].values  # set y as ocean temp C

x = x.reshape(-1, 1)  # Reshape to avoid value error
y = y.reshape(-1, 1)

# Deal with missing data
from sklearn.preprocessing import Imputer

my_imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0)
my_imputer = my_imputer.fit(x)
x = my_imputer.transform(x)

my_imputer = my_imputer.fit(y)
y = my_imputer.transform(y)

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
corr = np.corrcoef(x_train.T, y_train.T)

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
plt.title('Salinity vs Ocean Temp (C) (Training set)')
plt.xlabel('Salinity')
plt.ylabel('Temp (C)')

# Visualize the test set
plt.figure(2)
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, regressor.predict(x_test), color='blue')
plt.title('Salinity vs Ocean Temp (C) (Test set)')
plt.xlabel('Salinity')
plt.ylabel('Temp (C)')

plt.show()
