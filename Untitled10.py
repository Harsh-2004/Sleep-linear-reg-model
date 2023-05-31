#!/usr/bin/env python
# coding: utf-8

# In[7]:


pip install xgboost


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from a CSV file
df = pd.read_csv('sleep_newww - Sheet1.csv')

# Separate the features (X) and labels (y)
X = df[['Sleep duration', 'Exercise frequency']]
y = df['Sleep effIicency 1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy:", rf_accuracy)

x1 = np.linspace(X['Sleep duration'].min() - 2, X['Sleep duration'].max() + 2, 100)
x2 = np.linspace(X['Exercise frequency'].min() - 2, X['Exercise frequency'].max() + 2, 100)
xx1, xx2 = np.meshgrid(x1, x2)
Z = rf_model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.figure()
plt.contourf(xx1, xx2, Z, alpha=0.3)
plt.scatter(X['Sleep duration'], X['Exercise frequency'], c=y, edgecolors='k')
plt.xlabel('Sleep duration')
plt.ylabel('Exercise frequency')
plt.title('Decision Boundary (Random Forest)')
plt.xlim(X['Sleep duration'].min() - 2, X['Sleep duration'].max() + 2)
plt.ylim(X['Exercise frequency'].min() - 2, X['Exercise frequency'].max() + 2)
plt.show()


# In[ ]:




