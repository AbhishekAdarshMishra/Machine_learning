# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:09:27 2019

@author: KIIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values #ilco is use to select and [row,coloumn] (:)mns all and (:-1) mns all -1 i.e leaving last coloumn
y=dataset.iloc[:,2].values# 3 mns 3rd coloumn and index start from 0

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
  
 

#fitting Poly regression to the dataset
 from sklearn.preprocessing import PolynomialFeatures
 poly_reg=PolynomialFeatures(degree= 4) # change 2 to 3 to 4 for best possible case
 x_poly=poly_reg.fit_transform(x)
 poly_reg.fit(x_poly,y)
 lin_reg2=LinearRegression()
 lin_reg2.fit(x_poly,y)
 
 #visualising the Linear Regression results
 plt.scatter(x,y,color='red')
 plt.plot(x,lin_reg.predict(x),color='blue')
 plt.title("truth vs bluff(linear regression)")
 plt.xlabel("position level")
 plt.ylabel("salary")
 plt.show()
 
 
#visualising the polynomial regression results
 x_grid=np.arange(min(x),max(x),0.1)
 x_grid=x_grid.reshape((len(x_grid),1))#for more continuous line
 plt.scatter(x,y,color='red')
 plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')#see this line
 plt.title("truth vs bluff(poly regression)")
 plt.xlabel("position level")
 plt.ylabel("salary")
 plt.show()
 
 #predicting a new result with Linear regression
 
 lin_reg.predict(np.array(6.5).reshape(1,-1))
 
 
 
 #predicting a new result with poly regression
lin_reg2.predict(poly_reg.fit_transform(np.array(6.5).reshape(1,-1)))