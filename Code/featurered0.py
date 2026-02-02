#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 23:40:40 2025

@author: vuddameri
"""

# Install dependencies if needed:
# pip install pandas numpy scikit-learn pymrmr umap-learn matplotlib
import os
import tqdm
import pandas as pd
from umap import UMAP
import pymrmr
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

path = '/home/vuddameri/Ameya/Code' # copied from ameya/archive/greproc folder
os.chdir(path)
csv_file = "features2.csv"    # Path to your CSV file
label_column = "Label"           # Column name for labels
feature_counts = [20, 50, 100, 150, 200,250,300]  # mRMR feature counts to test
umap_dims = [2, 5, 10, 20,30]       # UMAP output dimensions to test

# ===== LOAD DATA =====
df = pd.read_csv(csv_file)
df = df.drop(columns=['Filename']) # remove the Filename column
y_raw = df[label_column]
X = df.drop(columns=[label_column]) # retain all columns except Labels

# ===== LABEL ENCODING =====
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Label mapping:", label_mapping)

# Create dataframe for pymrmr (label must be first column)
df_for_mrmr = pd.DataFrame(X.copy())
df_for_mrmr.insert(0, 'label', y)

# ===== MODEL & METRICS =====
f1_macro = make_scorer(f1_score, average='macro')
results = []

# ===== LOOP OVER SETTINGS =====
for f_count in feature_counts:
    selected = pymrmr.mRMR(df_for_mrmr, 'MIQ', f_count)
    X_sel = X[selected]
    
    for dim in umap_dims:
        umap_model = umap.UMAP(
            n_components=dim,
            random_state=42
        )
        # Unsupervised UMAP
        X_umap = umap_model.fit_transform(X_sel)
        # For supervised UMAP: X_umap = umap_model.fit_transform(X_sel, y)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        score = cross_val_score(model, X_umap, y, cv=5, scoring=f1_macro).mean()
        results.append((f_count, dim, score))

# ===== RESULTS TO DF =====
res_df = pd.DataFrame(results, columns=['mRMR_features', 'UMAP_dims', 'F1_macro'])

# ===== PLOT PERFORMANCE CURVES =====
plt.figure(figsize=(8,6))
for dim in umap_dims:
    subset = res_df[res_df['UMAP_dims'] == dim]
    plt.plot(subset['mRMR_features'], subset['F1_macro'], marker='o', label=f'UMAP {dim}D')
plt.xlabel("Number of mRMR features")
plt.ylabel("Macro F1-score")
plt.title("mRMR + UMAP tuning (multi-class target)")
plt.legend()
plt.show()

# ===== SAVE LABEL MAPPING =====
mapping_df = pd.DataFrame(list(label_mapping.items()), columns=['Class_Name', 'Encoded_Label'])
mapping_df.to_csv("label_mapping.csv", index=False)
print("Saved label mapping to label_mapping.csv")
