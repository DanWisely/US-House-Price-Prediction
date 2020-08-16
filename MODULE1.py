#!/usr/bin/env python
# coding: utf-8

# # as MODULE1.py - Regression Models

# In[1]:


# Import necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load EDA.csv file that has been preprocessed 

url="C:\\Users\\donte\\Desktop\\AIAP Test\\data\\EDA_forML.csv"
df = pd.read_csv(url)
df.head() 


# In[3]:


df.shape


# In[4]:


# Set up dataset for train-test-Split


# In[5]:


y = df['price']
X = df.iloc[:,1:] # set X indept variables; take all rows, all cols except 1st col
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

## => train data 80%, 15750 vs test data 20%, 3938, both across 11 cols


# In[6]:


X.head()


# In[7]:


y.head()


# In[8]:


# Model Model Build & Evaluation


# In[9]:


# 1st Approach Regression Model (Linear vs Lasso vs Ridge)


# In[10]:


### 1st Regression Model: Linear Regression


# In[11]:


lm = linear_model.LinearRegression()
lm_model = lm.fit(X_train, y_train)
lm_y_pred = lm.predict(X_test)

lm_y_pred
## the predicted hse price vs hse listed in test.data


# In[12]:


y_test.values


# In[13]:


plt.scatter(y_test, lm_y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# In[14]:


print('Score', lm_model.score(X_test, y_test))


# In[15]:


### 2nd Regression Model: Lasso Regression


# In[16]:


ls = linear_model.Lasso(max_iter=2000)
ls_model = ls.fit(X_train, y_train)
ls_y_pred = ls.predict(X_test)

ls_y_pred
## => the predicted hse price vs hse listed in test.data


# In[17]:


y_test.values


# In[18]:


plt.scatter(y_test, ls_y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# In[19]:


print('Score', ls_model.score(X_test, y_test))


# In[20]:


### 3rd Regression Model: Ridge Regression


# In[21]:


lr = linear_model.Ridge(max_iter=2000)
lr_model = lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)

lr_y_pred
## => the predicted hse price vs hse listed in test.data


# In[22]:


y_test.values


# In[23]:


plt.scatter(y_test, lr_y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# In[24]:


print('Score', lr_model.score(X_test, y_test))


# In[25]:


# Scoring - Regression Model
print('Linear Regression score is %f' % lm.score(X_test, y_test))
print('Ridge score is %f' % ls.score(X_test, y_test))
print('Lasso score is %f' % lr.score(X_test, y_test))


# In[26]:


## Summary of Model Performance


# In[27]:


#### All 3 Model seemed to have Overfitting issue, scoring 1, need to further augment training data to make Model 
#### more genralise for better performance
#### Linear Regression aim to reduce Cost Function (Error), Lasso & Ridge should be better as Cost Fn also added Lamda & slope


# In[ ]:




