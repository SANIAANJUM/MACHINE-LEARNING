#!/usr/bin/env python
# coding: utf-8


# In[1]:


import numpy as np 
# linear algebra
import pandas as pd 
# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# In[2]:


import os



# In[3]:


print(os.listdir("../python"))



# In[4]:


train_df = pd.read_csv('../python/train.csv')
test_df = pd.read_csv('../python/test.csv')



# In[5]:


train_df.head()


# In[6]:


# data size
train_df.shape



# In[7]:


digits = train_df.drop(['label'], 1).values
digits = digits / 255
label = train_df['label'].values



# In[8]:


# Show 25 digits of data
fig, axis = plt.subplots(5, 5, figsize=(22, 20))

for i, ax in enumerate(axis.flat):
    ax.imshow(digits[i].reshape(28, 28), cmap='binary')
    ax.set(title = "Real digit is {}".format(label[i]))



# In[9]:


# Machine Learning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



# In[10]:


digits.shape



# In[11]:


# Set X, y for fiting
X = digits
y = label
#X_test = test_df.values # file data



# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[14]:


from sklearn.ensemble import RandomForestClassifier


# Seting our model
model = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test) 
# predict our file test data
print("Model accuracy is: {0:.3f}%".format(accuracy_score(y_test, y_pred) * 100))



# In[15]:


# Compare our result
fig, axis = plt.subplots(5, 5, figsize=(18, 20))

for i, ax in enumerate(axis.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='binary')
    ax.set(title = "Predicted digit {0}\nTrue digit {1}".format(y_pred[i], y_test[i]))



# In[16]:


test_X = test_df.values
rfc_pred = model.predict(test_X)
sub = pd.read_csv('../python/sample.csv')
sub.head()



# In[17]:


# Make submission file
sub['Label'] = rfc_pred
sub.to_csv('submission.csv', index=False)
# Show our submission file
sub.head(10)


# In[ ]:







'''remove hash to implement'''