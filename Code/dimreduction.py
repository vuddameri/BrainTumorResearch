#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 01:17:38 2025

@author: vuddameri
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import pymrmr
import pacmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm import tqdm

# Load data and preprocessing (same as before)
path = '/home/vuddameri/Ameya/Code' # copied from ameya/archive/greproc folder
os.chdir(path)
csv_file = "features2.csv"    # Path to your CSV file
labelcol = "Label"           # Column name for labels
df = pd.read_csv(csv_file)
X = df.drop(columns=[labelcol,"Filename"])
feature_names = X.columns.to_list()
Y = df[[labelcol]].values
encoder = OrdinalEncoder()
df[[labelcol]] = encoder.fit_transform(Y)
labels = df[labelcol].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

def discretize_features(df, n_bins=10):
    df_disc = df.copy()
    for col in df.columns:
        df_disc[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
    return df_disc

X_disc = discretize_features(X_scaled_df)
data_for_mrmr = X_disc.copy()
data_for_mrmr[labelcol] = labels

# Run mRMR once to get full ranking of features
print("Running mRMR on full feature set for ranking...")
max_features = min(100, len(feature_names))  # max features to consider
all_selected_features = pymrmr.mRMR(data_for_mrmr, 'MIQ', max_features)

print("Starting feature number search with cross-validation...")

embedding_dim = 8
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

# Try different numbers of features
for k in tqdm(range(10, max_features+1, 5), desc='Feature count'):
    selected_features = all_selected_features[:k]
    X_selected = X_scaled_df[selected_features].values

    # PACMAP embedding
    embedder = pacmap.PaCMAP(n_components=embedding_dim, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    X_embedded = embedder.fit_transform(X_selected)

    # Cross-validate classifier on embedded features
    scores = cross_val_score(clf, X_embedded, labels, cv=cv, scoring='accuracy', n_jobs=-1)
    mean_score = np.mean(scores)

    results.append((k, mean_score))
    print(f"Features: {k}  |  CV Accuracy: {mean_score:.4f}")

# Find the best k
best_k, best_score = max(results, key=lambda x: x[1])
print(f"\nOptimal number of features: {best_k} with CV accuracy: {best_score:.4f}")
