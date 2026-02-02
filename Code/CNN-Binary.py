#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 01:50:43 2025

@author: vuddameri
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import os

# --------------------------
# 1) Load your dataset
# Replace this line with your actual data loading
# df = pd.read_csv("your_features.csv")

# Labels are 0,1,2,3 where 2 = no tumor
# Convert to binary: 2 -> 0 (no tumor), others -> 1 (tumor)
df['binary_label'] = (df['label'] != 2).astype(int)

# Feature columns (adjust if different)
feature_cols = [f"f{i}" for i in range(12)]

X = df[feature_cols].values.astype(np.float32)
y = df['binary_label'].values.astype(np.int64)

# --------------------------
# 2) Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --------------------------
# 3) Train/Val/Test split
SEED = 42
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED
)

# Reshape for Conv1D: (samples, timesteps, channels)
X_train_cnn = np.expand_dims(X_train, axis=2)
X_val_cnn   = np.expand_dims(X_val, axis=2)
X_test_cnn  = np.expand_dims(X_test, axis=2)

# --------------------------
# 4) Compute class weights for imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
cw_dict = {i: w for i, w in enumerate(class_weights)}

# --------------------------
# 5) Model Components
def squeeze_excite_block(inputs, ratio=4):
    filters = inputs.shape[-1]
    se = tf.reduce_mean(inputs, axis=1)  # Global avg pool (batch, channels)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.Multiply()([inputs, tf.expand_dims(se, 1)])
    return se

def residual_conv_block(x, filters, kernel_size=3):
    shortcut = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    x = squeeze_excite_block(x)
    return x

# --------------------------
# 6) Build model
inputs = tf.keras.Input(shape=(X_train.shape[1], 1))
x = residual_conv_block(inputs, 32, 3)
x = residual_conv_block(x, 64, 3)
x = residual_conv_block(x, 96, 3)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# --------------------------
# 7) Compile model with metrics
METRICS = [
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=METRICS
)

# --------------------------
# 8) Callbacks
checkpoint_path = "best_tabular_cnn_tf.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_auc', mode='max'),
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', patience=4, factor=0.5, verbose=1)
]

# --------------------------
# 9) Train
history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=40,
    batch_size=64,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=2
)

# --------------------------
# 10) Evaluate on test set
test_results = model.evaluate(X_test_cnn, y_test, verbose=0)
print("Test metrics:", dict(zip(model.metrics_names, test_results)))

# --------------------------
# 11) Compute F1 on test set
y_pred_probs = model.predict(X_test_cnn).ravel()
y_pred_labels = (y_pred_probs >= 0.5).astype(int)
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred_labels)
print("Test F1 score:", f1)
