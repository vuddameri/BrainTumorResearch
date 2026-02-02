#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 01:19:53 2025

@author: vuddameri
"""
# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

path = '/home/vuddameri/Ameya/Code'
fname = 'transdata.csv'
# 1. Read CSV file
os.chdir(path)
df = pd.read_csv(fname)

# 2. Define features and target
X = df.drop(columns=['y','Label','FileName'])  # replace with your target column name
Y = [0 if lab == 'notumor' else 1 for lab in df.Label] # create Binary data

# 3. Train-test split - 70% 30% split 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# 5. Fit the Naive Bayes Regression Model
gnb = GaussianNB()
gnb.fit(X_train,y_train)

print("Classes:", gnb.classes_)
print("Class priors:", gnb.class_prior_)
print("Class counts:", gnb.class_count_)
print("\nFeature means per class:\n", pd.DataFrame(gnb.theta_, index=gnb.classes_, columns=X_train.columns))
print("\nFeature variances per class:\n", pd.DataFrame(gnb.var_, index=gnb.classes_, columns=X_train.columns))

# 6. Predictions and ROC metrics
y_pred_prob = gnb.predict(X_test)

# ROC AUC
auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC AUC: {auc:.3f}")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Selection of optimal threshold

cutoffs = np.arange(0.0, 1.01, 0.01)
scores = []

for c in cutoffs:
    y_pred_class = (y_pred_prob >= c).astype(int)
    acc = accuracy_score(y_test, y_pred_class)
    scores.append((c, acc))
# Find best cutoff by accuracy
best_cutoff, best_acc = max(scores, key=lambda x: x[1])
print(f"Best cutoff: {best_cutoff:.2f}, Accuracy: {best_acc:.3f}")

#  classification at given threshold
thresh  = 0.01
y_pred_class = (y_pred_prob >= thresh).astype(int)
print(confusion_matrix(y_test, y_pred_class))
print(classification_report(y_test, y_pred_class))
