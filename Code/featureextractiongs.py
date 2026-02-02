#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 22:46:50 2025

@author: vuddameri
"""

import os
import cv2
import numpy as np
import pandas as pd
import mahotas
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# ==== UPDATE THIS PATH ====
data_dir = "/home/vuddameri/Ameya/archive/All"  # Root folder containing the 4 class folders

# Your 4 class folders:Make sure the names match if changed
labels = ["glioma", "meningioma", "pituitary", "notumor"] 
label_map = {label: idx for idx, label in enumerate(labels)}  # Map to 0,1,2,3; Dictionary

# LBP params
radius = 1
n_points = 8 * radius

def preprocess_gray(img_path, size=(128, 128)):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    gray_blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)
    return gray_blurred

def gabor_features(img_gray):
    features = {}
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)
    for i, kernel in enumerate(kernels):
        filtered = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel)
        features[f'gabor_mean_{i}'] = filtered.mean()
        features[f'gabor_std_{i}'] = filtered.std()
    return features

def canny_edge_features(img_gray):
    features = {}
    edges = cv2.Canny(img_gray, 100, 200)
    features['edge_pixel_count'] = np.sum(edges > 0)
    features['edge_pixel_ratio'] = features['edge_pixel_count'] / edges.size
    return features

def zernike_moments(img_gray, radius=64, degree=8):
    features = {}
    img_resized = cv2.resize(img_gray, (2*radius, 2*radius))
    _, img_bin = cv2.threshold(img_resized, 0, 255, cv2.THRESH_OTSU)
    img_bin = img_bin // 255
    moments = mahotas.features.zernike_moments(img_bin, radius, degree)
    for i, val in enumerate(moments):
        features[f'zernike_{i}'] = val
    return features
# exract HOG and GLCM, LBP, Hist, ...
def extract_gray_features(img_gray):
    features = {}
    features['entropy'] = shannon_entropy(img_gray)
    features['mean_gray'] = np.mean(img_gray)
    features['std_gray'] = np.std(img_gray)
    features['skew_gray'] = skew(img_gray.flatten())
    features['kurtosis_gray'] = kurtosis(img_gray.flatten())

    hog_feats = hog(img_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    for i, val in enumerate(hog_feats[:50]):
        features[f'hog_{i}'] = val

    glcm = greycomatrix(img_gray, [1], [0], levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']:
        features[f'glcm_{prop}'] = greycoprops(glcm, prop)[0, 0]

    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    for i, val in enumerate(hist):
        features[f'lbp_{i}'] = val

    moments = cv2.moments(img_gray)
    huMoments = cv2.HuMoments(moments).flatten()
    for i, val in enumerate(huMoments):
        features[f'humoment_{i}'] = val

    features.update(gabor_features(img_gray))
    features.update(canny_edge_features(img_gray))
    features.update(zernike_moments(img_gray))

    return features

data = []

for label in labels:
    folder = os.path.join(data_dir, label)
    for fname in tqdm(os.listdir(folder), desc=f"Processing {label}"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            fpath = os.path.join(folder, fname)
            img_gray = preprocess_gray(fpath)
            feats = extract_gray_features(img_gray)
            feats['label'] = label
            feats['label_num'] = label_map[label]
            feats['filename'] = fname
            data.append(feats)

df = pd.DataFrame(data)
df.to_csv("mri_gray_features_full_4classes.csv", index=False)
print("✅ Feature extraction done and saved to 'mri_gray_features_full_4classes.csv'")
