#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class Adaboost:
    def __init__(self, n_clf):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []
        
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("inf")
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


# In[9]:


def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


# In[10]:


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


# In[16]:


clf = Adaboost(n_clf=50)
clf.fit(X, targ)
y_pred = clf.predict(x_test)
acc = accuracy(target, y_pred)
print("Accuracy:", acc)


# In[ ]:




