#!/usr/bin/env python
# coding: utf-8

# # as MODULE2.py - Classification Models

# In[1]:


# Import necessary Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error


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


X = df.iloc[:,1:] # set X indept variables; take all rows, all cols except 1st col
y = df.iloc[:,0]  # set y dept tgt; take all rows, take 1st col only


# In[6]:


X.head()


# In[7]:


y.head()


# In[8]:


# Peform a train/test split to obtain a subset of data to test the models. The split used is a 80%/20%.


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

## => train data 80%, 15750 vs test data 20%, 3938, both across 11 cols


# In[10]:


# 2nd Approach Classification Model (Random Forest vs Xgboost vs ANN - Artificial Neural Network)


# In[11]:


### 1st Classification Model: Random Forest 


# In[12]:


### Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFclassifier=RandomForestClassifier()
RFmodel=RFclassifier.fit(X_train,y_train)
RF_y_pred=RFclassifier.predict(X_test)

RF_y_pred
## => the predicted hse price vs hse listed in test.data


# In[13]:


y_test.values
## => the actual hse price


# In[14]:


from sklearn.metrics import accuracy_score
print("Accuracy_score: {}".format(accuracy_score(y_test,RF_y_pred)))


# In[15]:


### 2nd Classification Model: Xgboost


# In[16]:


import xgboost
XGBclassifier=xgboost.XGBRegressor()
XGBmodel=XGBclassifier.fit(X_train,y_train)


# In[17]:


XGB_y_pred=XGBclassifier.predict(X_test)
XGB_y_pred

## => the predicted hse price vs hse listed in test.data


# In[18]:


y_test.values
## => the actual hse price


# In[19]:


### 3rd Classification Model: ANN - Artificial Neural Network


# In[20]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

# Initialising the ANN
ANNclassifier = Sequential()

# Adding the input layer and the first hidden layer
ANNclassifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu',input_dim = 11)) # 11 indept features

# Adding the second hidden layer
ANNclassifier.add(Dense(output_dim = 25, init = 'he_uniform',activation='relu'))

# Adding the third hidden layer
ANNclassifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu'))
# Adding the output layer
ANNclassifier.add(Dense(output_dim = 1, init = 'he_uniform')) #solving Regression type problem, don't use activation on last layer
                                                        # usually we a sigmoid fn in last layer for Classification type problem
                                                        # output_dim 1 for "price" prediction

# Compiling the ANN
ANNclassifier.compile(loss= 'mean_squared_error', metrics=['accuracy'], optimizer='Adamax')

# verify the network structure
ANNclassifier.summary()


# In[21]:


# Fitting the ANN to the Training set
model_history=ANNclassifier.fit(X_train.values, y_train.values,validation_split=0.20, batch_size = 11, nb_epoch = 200)


# In[22]:


ANNy_pred=ANNclassifier.predict(X_test)
ANNy_pred
## => the predicted hse price vs hse listed in test.data


# In[23]:


y_test.values
## => the actual hse price in test data


# In[24]:


from sklearn.metrics import accuracy_score, roc_auc_score
# Scoring - Regression Model
print('Random Forest score is %f' % RFmodel.score(X_test, y_test))
print('Xgboost score is %f' % XGBmodel.score(X_test, y_test))


# In[ ]:





# In[ ]:





# In[ ]:




