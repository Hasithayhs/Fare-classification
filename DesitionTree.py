#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
ride_data = pd.read_csv("C:/Users/uamarh1/Pictures/train.csv")

ride_data.head()


# In[2]:


ride_data=ride_data.dropna()

ride_data.head(20)

ride_data = ride_data.iloc[:,~ride_data.columns.isin(['tripid'])]
ride_data = ride_data.iloc[:,~ride_data.columns.isin(['drop_time'])]
ride_data = ride_data.iloc[:,~ride_data.columns.isin(['pickup_time'])]

ride_data.head()

X = ride_data.drop('label', axis=1)
Y = ride_data['label']
Y.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

X_train = scaler.transform(X)

Y = Y.replace(to_replace=['correct', 'incorrect'], value=[1, 0])

Y.head()


# In[4]:


from sklearn.metrics import confusion_matrix 
#from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


# In[13]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 
clf_gini_new=clf_gini.fit(X_train, Y)


# In[6]:


test_data = pd.read_csv("C:/Users/uamarh1/Pictures/test.csv")
test_data = test_data.iloc[:,~test_data.columns.isin(['tripid'])]
test_data = test_data.iloc[:,~test_data.columns.isin(['drop_time'])]
test_data = test_data.iloc[:,~test_data.columns.isin(['pickup_time'])]


# In[15]:


y_pred= clf_gini_new.predict(test_data)


# In[17]:


final = pd.DataFrame(data=y_pred, columns=["prediction"])


# In[18]:


final.head()


# In[20]:


final.to_csv(r'C:/Users/uamarh1/Pictures/final.csv')


# In[ ]:


clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
clf_entropy.fit(X_train, y_train)
y_pred_entro= clf_entropy.predict(test_data)

    

