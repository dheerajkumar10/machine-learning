#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[131]:


#!/usr/bin/env python
# coding: utf-8

# In[12]:


from math import cos, sin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[35]:

data = pd.read_csv("proj1-4.csv")
df = pd.DataFrame(data)
df["X"] = df.X.str.replace("(", "").str.strip()
df["X"] = df.X.str.replace(")", "").str.strip()
df["Y"] = df.Y.str.replace(")", "").str.strip()
df["X"] = df["X"].astype(float)
df["Y"] = df["Y"].astype(float)


x = np.array(df["X"]).reshape(-1, 1)
y = np.array(df["Y"]).reshape(-1, 1)

plt.figure(figsize=[6, 6])
n = np.size(x)
plt.scatter(
    x=x,
    y=y,
)
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()

xmean, ymean = np.mean(x), np.mean(y)
SS_xy = np.sum(y * x) - n * ymean * xmean
SS_xx = np.sum(x * x) - n * xmean * xmean

b1 = SS_xy / SS_xx
b0 = ymean - b1 * xmean

print("Coefficient b1 is: ", b1)
print("Coefficient b0 is: ", b0)

plt.scatter(x, y, color="b", marker=".", s=100)
y_pred = b0 + b1 * x
# b2*sin(8*x) + b3*cos(8*x) + b4*sin(16*x) + b5*cos(16*x)+ b6*sin(24*x) + b7*cos(24*x)+ b8*sin(32*x) + b9*cos(32*x)+ b10*sin(40*x) + b11*cos(40*x)+ b12*sin(48*x) + b13*cos(48*x)
plt.plot(x, y_pred, color="r")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


line2 = np.full([n, 1], [ymean])
plt.scatter(x, y)
plt.plot(x, line2, c="g")
plt.show()


# In[90]:
def loss(y, y_h):

    loss = np.mean((y_h - y) ** 2)
    return loss


differences_line1 = y_pred - y
print("Loss", loss(y, y_pred))
line1sum = 0
for i in differences_line1:
    line1sum = line1sum + (i * i)
print(line1sum)

differences_line2 = line2 - y
line2sum = 0
for i in differences_line2:
    line2sum = line2sum + (i * i)
print(line2sum)
diff = line2sum - line1sum
rsquared = diff / line2sum
print("R-Squared is : ", rsquared)


# In[ ]:
def gradients(X, y, y_h):

    m = X.shape[0]
    dw = (1 / m) * np.dot(X.T, (y_h - y))
    db = (1 / m) * np.sum((y_h - y))

    return dw, db

def x_transform(X, degree):
    r = np.array(X)
    for i in range(1, degree + 1):
        m = np.append(np.vstack(list(map(lambda x: cos(6 * i * x), X))), np.vstack(list(map(lambda x: sin(6 * i * x), X))), axis=1)
        r = np.append(r, m, axis=1)

    r = np.append(X, r, axis=1)
    # m = np.append(np.vstack(list(map(lambda x: sin(6 * i * x), X))), np.vstack(list(map(lambda x: cos(6 * i * x), X))), axis=1)
    # n.append(m)
    # p = np.append((X,n), axis=1)
    return r


def train(X, y, bs, epochs, lr, degree):

    x = x_transform(X, degree)
    m, n = x.shape
    print("degree", degree)
    print("number of training examples: ", m)
    print("number of features: ", n)
    w = np.zeros((n, 1))
    b = 0
    y = y.reshape(m, 1)
    losses = []

    for epoch in range(epochs):
        for i in range((m - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x[start_i:end_i]
            yb = y[start_i:end_i]

            y_h = np.dot(xb, w) + b
            dw, db = gradients(xb, yb, y_h)
            w -= lr * dw
            b -= lr * db

        l = loss(y, np.dot(x, w) + b)
        losses.append(l)

    return w, b, losses


# w, b, l = train(x, y, bs=100, epochs=1000, lr=0.01)
# print(w)
# print(b)
# print(l)


def predict(X, w, b, degree):
    x1 = x_transform(X, degree)
    return np.dot(x1, w) + b

degree = 6
for i in range(1, degree + 1):
    w, b, l = train(x, y, bs=100, epochs=800, lr=0.01, degree=i)
    y_predd = predict(x, w, b, i)
    # print("Loss  for degree ", i, " is ",l )
    fig = plt.figure(figsize=(8, 6))
    plt.plot(x, y, "y.")
    print("loss {}".format(loss(y, y_predd)))
    plt.plot(x, y_predd, "r.")
    plt.legend(["Data", "Polynomial predictions"])
    plt.xlabel("X - Input")
    plt.ylabel("y - target / true")
    plt.title("Polynomial Regression")
    plt.show()


# In[132]:


data1 = pd.read_csv("C:/Users/dheer/OneDrive/Desktop/testdata.csv")
df2 = pd.DataFrame(data1)
df2["TX"] = df2["X"].astype(float)
df2["TY"] = df2["Y"].astype(float)
tx= np.array(df2["TX"]).reshape(-1, 1)
ty= np.array(df2["TY"]).reshape(-1, 1)
m= np.size(tx)


old_value = 4
def predict1(X, w, b, degree):
    global old_value
    x1 = x_transform(X, degree)
    w = w[:old_value]
    print(old_value)
    old_value = old_value+2
    return np.dot(x1, w) + b


degree = 6
for i in range(1, degree + 1):
    y_predd = predict1(tx, w, b, i)
    # print("Loss  for degree ", i, " is ",l )
    fig = plt.figure(figsize=(8, 6))
    plt.plot(tx,ty, "y.")
    print("loss {}".format(loss(ty, y_predd)))
    l[i] =loss(ty, y_predd)
    plt.plot(tx, y_predd, "r.")
    plt.legend(["Data", "Polynomial predictions"])
    plt.xlabel("X -test")
    plt.ylabel("test-y - target / true")
    plt.title("Polynomial Regression")
    plt.show()






# In[ ]:




