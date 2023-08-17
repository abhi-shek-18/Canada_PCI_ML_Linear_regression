#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[15]:


#loading our csv file
df=pd.read_csv("canada_per_capita_income.csv")
df.head()


# In[16]:


#plotting the scatter plot 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("Year")
plt.ylabel("Per_Capita_income(US$)")
plt.scatter(df.year,df.per_capita_income)


# In[17]:


#now first creating linear regression object
reg=linear_model.LinearRegression()


# In[18]:


#now fitting our dataframe
reg.fit(df[['year']],df.per_capita_income)


# In[19]:


reg.predict([[2020]])


# In[20]:


#reading the years csv file to predict the per capita income
yr=pd.read_csv("years.csv")


# In[21]:


#applying the prediction algorithm on the yr data frame
predict=reg.predict(yr)


# In[25]:


#create a new column in years csv file to display the predicted value of each year
yr['Per_Capita_Income']=predict
yr


# In[23]:


#saving the output to a new csv file
yr.to_csv('Predicted_CPI.csv',index=False)


# In[24]:


#plotting a linear model of the given input
plt.xlabel("year")
plt.ylabel("Per capita income (US$)")
plt.title("Canada per Capital Income")
plt.scatter(df.year,df.per_capita_income)
plt.plot(df.year,reg.predict(df[['year']]),color='red')
plt.show(block=True)


# In[ ]:




