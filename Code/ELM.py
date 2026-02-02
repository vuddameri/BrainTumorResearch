#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 02:27:58 2025

@author: vuddameri
"""
import os
import numpy as np
import pandas as pd
from skelm import ELMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
# --------------------------
# Step 1: Load your dataset
# --------------------------
path = '/home/vuddameri/Ameya/Code'
fname = 'transdata.csv'
# 1. Read CSV file
os.chdir(path)
df = pd.read_csv(fname)
# 2. Define features and target
X = df.drop(columns=['y','Label','FileName'])  # replace with your target column name
Y = [0 if lab == 'notumor' else 1 for lab in df.Label] # create Binary data

# --------------------------
# Step 2: Split data
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)

# --------------------------
# Step 3: Suggest hidden nodes
# --------------------------
n_train_samples = X_train.shape[0]
n_features = X_train.shape[1]

# Heuristic: sqrt(n_samples * n_features)
n_hidden_start = int(np.sqrt(n_train_samples * n_features))

# Suggested range for CV: ±50%
hidden_nodes_range = [
    max(10, int(n_hidden_start*0.5)),  # avoid <10 nodes
    n_hidden_start,
    int(n_hidden_start*1.5)
]
print("Hidden nodes to try:", hidden_nodes_range)

# --------------------------
# Step 4: Cross-validation to pick best hidden nodes
# --------------------------
best_score = 0
best_n_hidden = hidden_nodes_range[0]

for n in hidden_nodes_range:
    elm = ELMClassifier(n_neurons=n, ufunc='tanh', random_state=42)
    cv_score = cross_val_score(elm, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f"Hidden nodes: {n}, CV Accuracy: {cv_score:.4f}")
    if cv_score > best_score:
        best_score = cv_score
        best_n_hidden = n

print(f"Best number of hidden nodes: {best_n_hidden}, CV Accuracy: {best_score:.4f}")

# --------------------------
# Step 5: Train final model with best hidden nodes
# --------------------------
final_elm = ELMClassifier(n_hidden=best_n_hidden, activation='tanh', random_state=42)
final_elm.fit(X_train, y_train)

# --------------------------
# Step 6: Predictions
# --------------------------
y_pred = final_elm.predict(X_test)
y_prob = final_elm.predict_proba(X_test)[:, 1]  # probability for positive class

# --------------------------
# Step 7: Evaluation
# --------------------------
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.4f}")

# ROC AUC
auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC: {auc:.3f}")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

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
    y_pred_class = (y_prob >= c).astype(int)
    acc = accuracy_score(y_test, y_pred_class)
    scores.append((c, acc))
# Find best cutoff by accuracy
best_cutoff, best_acc = max(scores, key=lambda x: x[1])
print(f"Best cutoff: {best_cutoff:.2f}, Accuracy: {best_acc:.3f}")

#  classification at given threshold
thresh  = 0.5
y_pred_class = (y_prob >= thresh).astype(int)
print(confusion_matrix(y_test, y_pred_class))
print(classification_report(y_test, y_pred_class))
