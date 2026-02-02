#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 03:02:01 2025

@author: vuddameri
"""
import os
import numpy as np
import pandas as pd
from hpelm import ELM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from matplotlib import pyplot as plt

# --------------------------
# 1. Load dataset
# --------------------------
path = '/home/vuddameri/Ameya/Code'
fname = 'transdata.csv'
os.chdir(path)
df = pd.read_csv(fname)

X = df.drop(columns=['y', 'Label', 'FileName']).values
Y = np.array([0 if lab == 'notumor' else 1 for lab in df.Label])

# --------------------------
# 2. Split data: train + validation + test
# --------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# --------------------------
# 3. Hidden nodes range (heuristic)
# --------------------------
n_train_samples, n_features = X_train.shape
n_hidden_start = int(np.sqrt(n_train_samples * n_features))
hidden_nodes_range = [max(10, int(n_hidden_start*0.5)), n_hidden_start, int(n_hidden_start*1.5)]
print("Hidden nodes to try:", hidden_nodes_range)

# --------------------------
# 4. Manual "cross-validation" on validation set
# --------------------------
best_score = 0
best_n_hidden = hidden_nodes_range[0]

for n in hidden_nodes_range:
    elm = ELM(n_features, 1, classification="c")
    elm.add_neurons(n, "tanh")  # activation: tanh
    elm.train(X_train, y_train[:, None])
    
    y_val_prob = elm.predict(X_val)
    y_val_pred = (y_val_prob >= 0.5).astype(int).ravel()
    acc = accuracy_score(y_val, y_val_pred)
    print(f"Hidden nodes: {n}, Validation Accuracy: {acc:.4f}")
    if acc > best_score:
        best_score = acc
        best_n_hidden = n

print(f"Best hidden nodes: {best_n_hidden}, Validation Accuracy: {best_score:.4f}")

# --------------------------
# 5. Train final model on full training data
# --------------------------
final_elm = ELM(n_features, 1, classification="c")
final_elm.add_neurons(best_n_hidden, "tanh")
final_elm.train(X_train_full, y_train_full[:, None])

# --------------------------
# 6. Predictions on test set
# --------------------------
y_prob = final_elm.predict(X_test).ravel()
y_pred = (y_prob >= 0.49).astype(int)

# --------------------------
# 7. Evaluation
# --------------------------
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC: {auc:.3f}")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# --------------------------
# 8. Optimal threshold by accuracy
# --------------------------
cutoffs = np.arange(0.0, 1.01, 0.01)
scores = [(c, accuracy_score(y_test, (y_prob >= c).astype(int))) for c in cutoffs]
best_cutoff, best_acc = max(scores, key=lambda x: x[1])
print(f"Best cutoff: {best_cutoff:.2f}, Accuracy: {best_acc:.3f}")

