#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 00:20:20 2025

@author: vuddameri
"""

import os
import cv2
from tqdm import tqdm

# Update this to your original data folder path
data_dir = "/home/vuddameri/Ameya/archive/All"  

# Output directory in your home folder
output_base_dir = "/home/vuddameri/Ameya/archive/Grayproc"

labels = ["glioma", "meningioma", "pituitary", "notumor"]

def preprocess_and_save():
    for label in labels:
        input_folder = os.path.join(data_dir, label)
        output_folder = os.path.join(output_base_dir, label)
        os.makedirs(output_folder, exist_ok=True)

        for fname in tqdm(os.listdir(input_folder), desc=f"Processing {label}"):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, fname)
                img = cv2.imread(img_path)
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray_clahe = clahe.apply(gray)

                # Resize to 128x128
                resized = cv2.resize(gray_clahe, (128, 128))

                # Save output
                save_path = os.path.join(output_folder, fname)
                cv2.imwrite(save_path, resized)

    print(f"✅ Preprocessed images saved to {output_base_dir}")

preprocess_and_save()
