#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 00:06:14 2025

@author: vuddameri
"""

import os
import cv2
from tqdm import tqdm

data_dir = "/home/vuddameri/Ameya/archive/All"
gray_dir = "/home/vuddameri/Ameya/archive/Grey" # Create before running the code

labels = ["glioma", "meningioma", "pituitary", "notumor"]

def preprocess_and_save_gray():
    for label in labels:
        input_folder = os.path.join(data_dir, label)
        output_folder = os.path.join(gray_dir, label)
        os.makedirs(output_folder, exist_ok=True)
        for fname in tqdm(os.listdir(input_folder), desc=f"Processing {label}"):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, fname)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Optional: apply CLAHE or resizing here if you want
                cv2.imwrite(os.path.join(output_folder, fname), gray)

preprocess_and_save_gray()  # Call the function
print("✅ Grayscale images saved!") # Print once everythinfis done
