#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 22:17:30 2025

@author: vuddameri
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.metrics import roc_curve, roc_auc_score,f1_score
from matplotlib import pyplot as plt
path = '/home/vuddameri/Ameya/Code'
fname = 'transdata.csv'
# 1. Read CSV file
os.chdir(path)
df = pd.read_csv(fname)

# 2. Define features and target
X = df.drop(columns=['y','Label','FileName'])  # replace with your target column name
Y = [0 if lab == 'notumor' else 1 for lab in df.Label] # create Binary data
scaler = StandardScaler()
X_scl = scaler.fit_transform(X)
# 3. Train-test split - 70% 30% split 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Grid Search
# ---- Define parameter grid (logarithmic scale as suggested) ----
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

# ---- Set up SVM and GridSearch ----
svc = SVC(probability=True)
grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    scoring='accuracy',    # or 'f1', 'roc_auc' depending on your needs
    cv=5,                  # 5-fold cross-validation
    n_jobs=-1,             # use all CPU cores
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)

# ---- Best parameters and score ----
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
best_model = grid_search.best_estimator_

# ---------------------------
# 5. Predictions & Probabilities
# ---------------------------
y_prob = best_model.predict_proba(X_test_scaled)[:,1]  # probability of tumor
y_pred = best_model.predict(X_test_scaled)     # default 0.5 cutoff


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


thresholds = np.arange(0.01, 1.01, 0.01)
scores = []

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    scores.append(accuracy_score(y_pred, y_pred_t))  # minority = 0

optimal_idx = np.argmax(scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal cutoff : {optimal_threshold:.2f}, Accuracy={scores[optimal_idx]:.3f}")

#  classification at given threshold
thresh  = 0.53
y_pred_class = (y_prob >= thresh).astype(int)
print(confusion_matrix(y_test, y_pred_class))
print(classification_report(y_test, y_pred_class))

y_pred_prob = best_model.predict_proba(X_scl)
y_pred_class = (y_pred_prob[:,1] >= thresh).astype(int)
y_pred_class = list(y_pred_class)
zz = {'Label': df.Label, 'Y':df.y, 'Ybin':Y, 'Ypbin':y_pred_class,'y_prob': y_pred_prob[:,1],
      'Filename':df.FileName}
zzf = pd.DataFrame(zz)
zzf.to_csv('SVMRBFpred.csv',index=False)
