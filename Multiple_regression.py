# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:22:04 2019

@author: KIIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values #ilco is use to select and [row,coloumn] (:)mns all and (:-1) mns all -1 i.e leaving last coloumn
y=dataset.iloc[:,4].values# 3 mns 3rd coloumn and index start from 0


  from sklearn.preprocessing import LabelEncoder , OneHotEncoder
 labelencoder_x=LabelEncoder()
 x[:,3]=labelencoder_x.fit_transform(x[:,3])
 # but there is a problem as number encoded so it can under stand as one has heigher value than of other so we use dummy encoder by taking 1,0 for all label by onehotencoder 
 onehotencoder=OneHotEncoder(categorical_features=[3])
 x=onehotencoder.fit_transform(x).toarray()
 
 #avoiding dummy variable trap
 x=x[:,1:]
 
 
 
#Splitting the dataset into the training set and Test set
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)#20% test case wuth random taken the index 
 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


#building the optimal model using Backward Elimation
import statsmodels.formula.api as sn
# to add 1 at beginning for x0=1
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

# see if greatese p is less than 0.05 that is significant level than ok otherwise remove it and fiollow the steps again
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sn.OLS(endog=y,exog=x_opt).fit()

x_opt=x[:,[0,1,3,4,5]]
regressor_OLS=sn.OLS(endog=y,exog=x_opt).fit()

x_opt=x[:,[0,3,4,5]]
regressor_OLS=sn.OLS(endog=y,exog=x_opt).fit()

x_opt=x[:,[0,3,5]]
regressor_OLS=sn.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:,[0,3]]
regressor_OLS=sn.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()




#for automatic backward elimination with p-variable
"""import statsmodels.formula.api as sm
def backwardElimination(x,sl):
    numVars=len(x[0])
    for i in range(0,numVars):
        regressor_OLS = sm.OLS(y,x).fit()
        maxVar= max(regressor_OLS.pvalues).astype(float)
        if maxVar>sl:
            for j in range(0,numVars-i):
                if(regressor_OLS[j].pvalues[j].astype(float)==maxVar):
                    x=np.delete(x,j,1)
   regressor_OLS.summary()
   return x
SL=0.05
x_opt=x[:,[0,1,2,3,4,5]]
x_Modeled=backwardElimation(x_opt,SL)   """             
                
