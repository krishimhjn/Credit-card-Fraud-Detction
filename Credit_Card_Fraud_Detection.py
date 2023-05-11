#!/usr/bin/env python
# coding: utf-8

# In[33]:


# import libraries:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#Loading the dataset to a Pandas Dataframe

credit_card_data = pd.read_csv('creditcard.csv')
# let's see first 5 rows of the dataset:
credit_card_data.head(5)
# let's see last 5 rows of our dataset:
credit_card_data.tail()
# dataset information:
credit_card_data.info()
# checking number of missing values:
credit_card_data.isnull().sum()

# Find distribution of Normal transaction or Fraud transaction:
credit_card_data['Class'].value_counts()
# Separating the data:
normal = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
# check shape
print(normal.shape)
print(fraud.shape)

#visualize the data:
labels = ["Normal", "Fraud"]
count_classes = credit_card_data.value_counts(credit_card_data['Class'], sort=True)
count_classes.plot(kind="bar", rot=0)
plt.title("Project")
plt.ylabel("Count")
plt.xticks(range(2), labels)
# plt.show()

# statistical measures of the data:
normal.Amount.describe()
fraud.Amount.describe()

# visualize the data using seaborn:
sns.relplot(x='Amount', y='Time', hue='Class', data=credit_card_data)
plt.show()



# Compare values of both transactions:
credit_card_data.groupby('Class').mean()

# Now we will build a sample dataset containing similar distribution of normal transaction and fraud transaction:
normal_sample = normal.sample(n=492)
# Concat two data ( normal_sample and fraud) to create new dataframe which consist equal number of fraud transactions and normal transactions, In this way we balance our dataset (As our dataset is highly unbalanced initially) :
credit_card_new_data = pd.concat([normal_sample, fraud], axis=0)
# Let’s see our new dataset:
# credit_card_new_data

# Analyse our new dataset:
credit_card_new_data['Class'].value_counts()


# Splitting data into features and targets
X = credit_card_new_data.drop('Class', axis=1)
Y = credit_card_new_data['Class']

# splitting the data into training and testing data:
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state= 2)
print(X.shape, X_train.shape, X_test.shape)

# Creating Model:
model = LogisticRegression()
# training the Logistic Regression model with training data:
model.fit(X_train,Y_train)

# Model Evaluation
X_train_pred = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_pred, Y_train)
print('Accuracy of Training data:', training_data_accuracy)

# classification report of the model on training data:
print(classification_report(X_train_pred, Y_train))

# accuracy on test data:
X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, Y_test)
print('Accuracy of Testing data:', test_data_accuracy)

# confusion matrix and classification report of test data:
print(confusion_matrix(X_test_pred, Y_test))
print(classification_report(X_test_pred, Y_test))