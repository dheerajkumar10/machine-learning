#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import math


# In[2]:


for dirname, _, filenames in os.walk('C:/Users/dheer/OneDrive/Desktop/ML project'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


df = pd.read_csv('C:/Users/dheer/OneDrive/Desktop/ML project/train_images.csv', header = None)
df


# In[4]:


y = pd.read_csv('C:/Users/dheer/OneDrive/Desktop/ML project/train_labels.csv')
y


# In[5]:


y[y['Volcano?'] == 1]


# In[6]:


y['Volcano?'].value_counts()


# In[7]:


volcanos_index = y[y['Volcano?'] == 1].head().index
plt.title('Images with volcanos')
for i in volcanos_index:
    plt.imshow(df.iloc[i].values.reshape(110, 110, 1))
    plt.show()


# In[8]:


volcanos_index = y[y['Volcano?'] == 0].head().index
plt.title('Images without volcanos')
for i in volcanos_index:
    plt.imshow(df.iloc[i].values.reshape(110, 110, 1))
    plt.show()


# In[9]:


df_d = df.values.reshape(len(df), 110, 110, 1)
df_d = df_d / 255
print(f'Shape of each photo: {df_d[0].shape}')
print(f'Scaled data: \nMinimal value: {np.min(np.min(df_d))} \nMaximum value: {np.max(np.max(df_d))}')


# In[10]:


scaler= StandardScaler()
scaler.fit(df)


# In[11]:


scaled_data=scaler.transform(df)


# In[12]:


scaled_data


# In[13]:


def PCa(X , num_components):
     
    X_meaned = X - np.mean(X , axis = 0)
    cov_mat = np.cov(X_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced


# In[14]:


x_pca=PCa(scaled_data , 9)


# In[ ]:





# In[15]:


scaled_data.shape


# In[16]:


x_pca.shape


# In[17]:


y['Volcano?'][0]


# In[18]:


def euclidean_distance(row1,row2):
    distance=0.0;
    for i in range(len(row1)-1):
        distance = distance + (row1[i] - row2[i])**2
        sqrtdistance = math.sqrt(distance)
    return sqrtdistance


# In[19]:


def get_neighbours(train,test_row,k):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row,train_row)
        distances.append((train_row,dist))
    distances.sort(key=lambda tup:tup[1])
    neighbours = list()

    for i in range(k):
        neighbours.append(distances[i][0])
    return neighbours


# In[20]:


df1 = pd.read_csv('C:/Users/dheer/OneDrive/Desktop/ML project/test_images.csv', header = None)
df1


# In[21]:


scaler.fit(df1)


# In[22]:


scaled_data1=scaler.transform(df1)


# In[23]:


scaled_data1


# In[ ]:





# In[24]:


x_pca_test=PCa(scaled_data1 , 9)


# In[25]:


x_pca_test=x_pca_test.tolist()
x_pca=x_pca.tolist()
y_train=y['Volcano?'].tolist()


# In[26]:


outarray=[]
for i in range(len(x_pca_test)):
    volcano=0
    novolcano=0
    neighbours = get_neighbours(x_pca,x_pca_test[i],15)
    for neighbour in neighbours:
        l = x_pca.index(neighbour)
        if(y_train[l] == 0):
            novolcano = novolcano+1
        else:
            volcano = volcano+1
    if(novolcano >volcano):
        outarray.append(0)
    else:
        outarray.append(1)
         


# In[27]:


print(outarray[2])


# In[28]:


y_test1 = pd.read_csv('C:/Users/dheer/OneDrive/Desktop/ML project/test_labels.csv')


# In[29]:


y_test=y_test1['Volcano?'].tolist() 


# In[30]:


count=0
for i in range(len(y_test)):
    if(y_test[i]==outarray[i]):
        count=count+1
accuracy=count/(len(y_test))
print(accuracy)


# In[31]:


plt.hist(y["Volcano?"])
plt.show()


# In[32]:


for j in y_test1:
    X = y_test1[j]
    plt.figure()
    plt.hist(X)
    plt.xlabel(j)
plt.show()


# In[33]:


print('Images of volcanoes correctly predicted by our algorithm')
for i in range(len(y_test)):
    if ((outarray[i]==y_test[i]) and outarray[i]==1):
        plt.imshow(df1.iloc[i].values.reshape(110, 110, 1))
    plt.show()
        


# In[34]:


print(outarray)


# In[ ]:




