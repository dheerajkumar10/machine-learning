#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[55]:


data = pd.read_csv("proj1.csv")
data1 = pd.read_csv("testdata.csv")
df = pd.DataFrame(data)
df2 = pd.DataFrame(data1)
df["X"] = df.X.str.replace("(", "").str.strip()
df["X"] = df.X.str.replace(")", "").str.strip()
df["Y"] = df.Y.str.replace(")", "").str.strip()
df["X"] = df["X"].astype(float)
df["Y"] = df["Y"].astype(float)
df2["TX"] = df2["X"].astype(float)
df2["TY"] = df2["Y"].astype(float)
print(df)
print(df2)
x = np.array(df["X"]).reshape(-1, 1)
y = np.array(df["Y"]).reshape(-1, 1)
tx= np.array(df2["TX"])
ty= np.array(df2["TY"])
m= np.size(tx)
plt.figure(figsize=[8, 6])
n = np.size(x)
plt.scatter(
    x=x,
    y=y,
)
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()
n = np.size(x)


# In[59]:


def weight(p, X, a): 
    m = X.shape[0] 
    w = np.mat(np.eye(m)) 
    for i in range(m): 
        xi = X[i] 
        d = (-2 * a * a) 
        w[i, i] = np.exp(np.dot((xi-p), (xi-p).T)/d) 
        
    return w


# In[60]:


def predict(X, y, p, a): 
    k = X.shape[0]     
    X1 = np.append(X, np.ones(k).reshape(k,1), axis=1) 
    p1 = np.array([p, 1]) 
    w = weight(p1, X1, a) 
    
    theta = np.linalg.pinv(X1.T*(w * X1))*(X1.T*(w * y))  
    prediction = np.dot(p1, theta) 
    return theta, prediction


# In[65]:


def predictions(X, y, a, n):
    X_test = np.linspace(-3, 3, n)  
    preds = [] 
    for point in X_test: 
        theta, pred = predict(X, y, point, a) 
        preds.append(pred)
    X_test = np.array(X_test).reshape(n,1)
    preds =  np.array(preds).reshape(n,1) 
    plt.plot(X, y, 'b.')
    plt.plot(X_test, preds, 'r.')
    plt.show()
    
predictions(x, y, 0.25, n)


# In[ ]:


def predictions(X, y, a, n):
    X_test = np.linspace(-3, 3, n)  
    preds = [] 
    for point in X_test: 
        theta, pred = predict(X, y, point, a) 
        preds.append(pred)
    X_test = np.array(X_test).reshape(n,1)
    preds =  np.array(preds).reshape(n,1) 
    plt.plot(tx, ty, 'b.')
    plt.plot(X_test, preds, 'r.')
    plt.show()
    
predictions(x, y, 0.25, n)


# In[ ]:




