# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:15:09 2019

@author: KIIT
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:,:-1].values #ilco is use to select and [row,coloumn] (:)mns all and (:-1) mns all -1 i.e leaving last coloumn
y=dataset.iloc[:,1].values# 3 mns 3rd coloumn and index start from 0


  
 
#Splitting the dataset into the training set and Test set
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)#20% test case wuth random taken the index 
 
 
 # fitting Simple linear Regression to the training set
 from sklearn.linear_model import LinearRegression
 regressor=LinearRegression()
 regressor.fit(x_train,y_train)
 ￼
 #predicting the Test set result
 y_pred=regressor.predict(x_test)
 
 
 #visualising the training set result
 plt.scatter(x_train,y_train,color='red')
 plt.plot(x_train,regressor.predict(x_train),color='blue')#see this line we have to use pred of x train not x test as we are plotting fir test
 plt.title("salary vs experience(Training set)")
 plt.xlabel("year of experience")
 plt.ylabel("salary")
 plt.show()
 
 
 #visualising the test set result
 plt.scatter(x_test,y_test,color='red')
 plt.plot(x_train,regressor.predict(x_train),color='blue')# pred line for x train and x test is same so no need to change it
 plt.title("salary vs experience(Test set)")
 plt.xlabel("year of experience")
 plt.ylabel("salary")
 plt.show()
 
 
 