# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 18:47:15 2022

@author: Fatemeh
"""

#Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#______________________________________________________________________________
#Importing the Boston House Price Dataset
house_price_dataset = sklearn.datasets.load_boston()
print(house_price_dataset)

#______________________________________________________________________________
# Loading the dataset to a Pandas DataFrame
house_price_dataframe = pd.DataFrame(house_price_dataset.data,
                                     columns = house_price_dataset.feature_names)
print(house_price_dataframe)

# Print First 5 rows of our DataFrame
head = house_price_dataframe.head()
print(head)

# add the target (price) column to the DataFrame
house_price_dataframe['price'] = house_price_dataset.target
print(house_price_dataframe)

# checking the number of rows and Columns in the data frame
shape = house_price_dataframe.shape
print(shape)

#______________________________________________________________________________
# check for missing values
missing_values = house_price_dataframe.isnull().sum()
print(missing_values)

# statistical measures of the dataset
describe = house_price_dataframe.describe()
print(describe)

#______________________________________________________________________________
#Understanding the correlation between various features in the dataset
correlation = house_price_dataframe.corr()
print(correlation)

# constructing a heatmap to nderstand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',
            annot=True, annot_kws={'size':8}, cmap='Blues')
#______________________________________________________________________________
#Splitting the data and Target
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']
print(X, Y)

#Splitting the data into Training data and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
print("X_train:","\n" , X_train,"\n"
      "X_test:","\n",X_test)

print("X.shape:",X.shape,"\n"
      "X_train.shape:",X_train.shape,"\n"
      "X_test.shape:",X_test.shape,"\n")

#______________________________________________________________________________
#Model Training
#XGBoost Regressor --> this is a type of decesion tree and ensemble model

# loading the model
model = XGBRegressor()

# training the model with X_train
model.fit(X_train, Y_train)

#______________________________________________________________________________
#Evaluation

# accuracy for prediction on training data
training_data_prediction = model.predict(X_train)
print(training_data_prediction,"\n")

# R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)
print("R squared error : ", score_1,"\n")

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)
print('Mean Absolute Error : ', score_2,"\n")


#Visualizing the actual Prices and predicted prices
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()


# accuracy for prediction on test data
test_data_prediction = model.predict(X_test)
print(test_data_prediction,"\n")

# R squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", score_1,"\n")

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)
print('Mean Absolute Error : ', score_2,"\n")