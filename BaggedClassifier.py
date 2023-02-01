#!/usr/bin/env python
# coding: utf-8

# In[158]:


import numpy as np
import pandas as pd

class DecisionTreeClassifier:
    
    def cal_entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def create_split(self, X, threshold):
        left_idx_value = np.argwhere(X <= threshold).flatten()
        right_idx_value = np.argwhere(X > threshold).flatten()
        return left_idx_value, right_idx_value

    def cal_information_gain(self, X, y, threshold):
        parent_loss = self.cal_entropy(y)
        left_idx_value, right_idx_value = self.create_split(X, threshold)
        n, n_left, n_right = len(y), len(left_idx_value), len(right_idx_value)

        if n_left == 0 or n_right == 0: 
            return 0
        
        child_loss = (n_left / n) * self.cal_entropy(y[left_idx_value]) + (n_right / n) * self.cal_entropy(y[right_idx_value])
        return parent_loss - child_loss
    
    def __init__(self, max_depth, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def finished(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples):
            return True
        return False

    def best_split(self, X, y, features):
        split = {'score':- 1, 'feat': None, 'thresh': None}
        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self.cal_information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh
        return split['feat'], split['thresh']
    
    def build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))
        
        if self.finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return NodeItem(value=most_common_Label)

        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self.best_split(X, y, rnd_feats)

        left_idx, right_idx = self.create_split(X[:, best_feat], best_thresh)
        left_child = self.build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self.build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return NodeItem(best_feat, best_thresh, left_child, right_child)
    
    def pre_order_traverse(self, x, node):
        if node.is_it_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.pre_order_traverse(x, node.left)
        return self.pre_order_traverse(x, node.right)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict(self, X):
        predictions = [self.pre_order_traverse(x, self.root) for x in X]
        return np.array(predictions)
    
class NodeItem:
    def __init__(self, feature_value=None, threshold_value=None, left_node=None, right_node=None, *, value=None):
        self.feature = feature_value
        self.threshold = threshold_value
        self.left = left_node
        self.right = right_node
        self.value = value
        
    def is_it_leaf(self):
        return self.value is not None


# In[159]:


class BaggedDecisonTreeClassifier:
    
    def __init__(self,num_of_bagged):
        self.num_of_bagged=num_of_bagged
        
    def fit(self,X,y):
        self.models=[]
        for i in range(self.num_of_bagged):
            indexs=np.random.choice(len(X),size=len(X))
            Xi=X[indexs]
            Yi=y[indexs]
            model=DecisionTreeClassifier(max_depth=8)
            model.fit(Xi,Yi)
            self.models.append(model)
            
    def predict(self,X):
        pred=np.zeros(len(X))
        for model in self.models:
            pred=pred+model.predict(X)
        return np.round(pred/self.num_of_bagged)
    
    def acc(self,y_true,y_pred):
        return np.mean(y_true==y_pred)


# In[160]:


df = pd.read_csv("Data2.csv")
target = np.array(list(df.G))
df_clean = df.drop(columns=['G'])
data = df_clean.to_numpy()
X, targ = np.array(data),np.array(target)

df2 = pd.read_csv("testdata.csv")
test_target = np.array(list(df2.G))
df2_clean = df2.drop(columns=['G'])
data2 = df2_clean.to_numpy()
x_test, target = np.array(data2),np.array(test_target)


# In[161]:


b=BaggedDecisonTreeClassifier(10)
b.fit(X,targ)


# In[162]:


y_pred=b.predict(x_test)


# In[163]:


b.acc(target,y_pred)


# In[ ]:




