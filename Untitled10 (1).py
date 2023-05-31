#!/usr/bin/env python
# coding: utf-8

# In[7]:


pip install xgboost


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from a CSV file
df = pd.read_csv('sleep_newww - Sheet1.csv')

# Separate the features (X) and labels (y)
X = df[['Sleep duration', 'Exercise frequency']]
y = df['Sleep effIicency 1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the decision boundary
x1 = np.linspace(X['Sleep duration'].min() - 1, X['Sleep duration'].max() + 1, 100)
x2 = np.linspace(X['Exercise frequency'].min() - 1, X['Exercise frequency'].max() + 1, 100)
xx1, xx2 = np.meshgrid(x1, x2)
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.figure()
plt.contourf(xx1, xx2, Z, alpha=0.3)
plt.scatter(X['Sleep duration'], X['Exercise frequency'], c=y, edgecolors='k')
plt.xlabel('Sleep duration')
plt.ylabel('Exercise frequency')
plt.title('Decision Boundary')
plt.show()


# In[ ]:




