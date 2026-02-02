#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 00:39:19 2025

@author: vuddameri
"""
import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import mahotas
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# Path to your preprocessed grayscale images folder
dataset_path = "/home/vuddameri/Ameya/archive/Grayproc"  # change this

# GLCM parameters
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0]

# Zernike moment parameters
ZERNIKE_RADIUS = 21
ZERNIKE_DEGREE = 8

features_list = []
labels_list = []
fnames = []
for class_label in tqdm(os.listdir(dataset_path), desc="Classes"):
    class_dir = os.path.join(dataset_path, class_label)
    if not os.path.isdir(class_dir):
        continue

    for fname in os.listdir(class_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        fnames.append(fname)
        fpath = os.path.join(class_dir, fname)
        # Load grayscale image (already grayscale & resized)
        image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

        # --- HOG features ---
        hog_features = hog(image,
        orientations=9,            # Keep orientations at 9 or reduce if you want
        pixels_per_cell=(32, 32),  # Larger cells → fewer features
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
        )

        # --- Zernike moments ---
        # Threshold to binary image for Zernike moments
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        zernike_features = mahotas.features.zernike_moments(binary, radius=ZERNIKE_RADIUS, degree=ZERNIKE_DEGREE)

        # --- GLCM features ---
        glcm = graycomatrix(image, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                            levels=256, symmetric=True, normed=True)
        glcm_props = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            glcm_props.append(graycoprops(glcm, prop)[0, 0])

        # --- Summary statistics ---
        pixels = image.flatten()
        mean_val = np.mean(pixels)
        std_val = np.std(pixels)
        skewness = skew(pixels)
        kurt = kurtosis(pixels)
        min_val = np.min(pixels)
        max_val = np.max(pixels)

        # --- Entropy ---
        entropy_val = shannon_entropy(image)

        # --- Combine features ---
        features = np.hstack([
            hog_features,
            zernike_features,
            glcm_props,
            [mean_val, std_val, skewness, kurt, min_val, max_val, entropy_val]
        ])

        features_list.append(features)
        labels_list.append(class_label)

# Create DataFrame with meaningful column names
num_hog = len(hog_features)
num_zernike = len(zernike_features)
num_glcm = len(glcm_props)

columns = (
    [f"HOG_{i}" for i in range(num_hog)] +
    [f"Zernike_{i}" for i in range(num_zernike)] +
    [f"GLCM_{name}" for name in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']] +
    ['Mean', 'Std', 'Skewness', 'Kurtosis', 'Min', 'Max', 'Entropy']
)

df = pd.DataFrame(features_list, columns=columns)
df['Label'] = labels_list
df['Filename'] = fnames

# Save to CSV
fname = '/home/vuddameri/Ameya/archive/Grayproc/features2.csv'
df.to_csv(fname, index=False)
print(f"Feature extraction complete! Saved to {fname}")
df.shape