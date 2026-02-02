#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 23:18:09 2025
@author: vuddameri
"""
# Processing brain tumor images
# https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

#Reading Data and Preprocessing

#load libraries
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Path to dataset
path = '/home/vuddameri/Ameya/archive/All'
os.chdir(path)

# Get classes (folders)
classes = os.listdir(path)
print("Classes:", classes)

# Randomly pick one image and check if greyscale
# Explore the naming conventioned used to pick one
fname = '/home/vuddameri/Ameya/archive/All/glioma/Te-gl_0014.jpg'
img = cv2.imread(fname)
# Check number of channels
if len(img.shape) < 3 or img.shape[2] == 1:
    print("The image is grayscale.")
else:
    print("The image is RGB")

# Base folder containing subfolders
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Pick a random image from each class and plot

labsel = []
imgsel = []
plt.figure(figsize=(10, 10))
for i, class_name in enumerate(classes):
    folder_path = os.path.join(path, class_name)
    files = os.listdir(folder_path)
    image_path = os.path.join(folder_path, random.choice(files))

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert for matplotlib rgb
    imgsel.append(img)
    labsel.append(class_name) # should be same as classes 
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.title(class_name)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Exploratory Data Analysis


# Resize all Target size for resizing
targsize = (128, 128)
# Resize images
img128 = [cv2.resize(img, targsize) for img in imgsel]
# Convert to greyscale
imggs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in img128] 

# Plot histogram

plt.figure(figsize=(12, 8))
for i, img in enumerate(imggs):
    plt.subplot(2, 2, i + 1)
    plt.hist(img.ravel(), bins=256, range=[0, 256], color='gray')
    plt.title(f'Grayscale Histogram - {classes[i]}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# PLaying with background removal
for i, img in enumerate(imggs):
    plt.subplot(2, 2, i + 1)

    # Mask out zero pixels
    thresh = 10
    pixels = img.ravel()
    pixels_no_bg = pixels[pixels > thresh]

    plt.hist(pixels_no_bg, bins=256, range=[1, 256], color='gray')
    plt.title(f'No BG - {classes[i]}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Perform Histogram equalization for contrast enhancement

# Assume imggs is your list of grayscale images
histeqimg = []
claheimg = []
for img in imggs:
    # Create a mask for pixels >= threshold
    mask = img >= thresh
   # For visualization, set pixels < threshold to 0 (black)
    img_masked = np.where(mask, img, 0).astype(np.uint8)    
    # Enhance contrast using histogram equalization on the masked image
    # Note: histogram equalization works on entire image, so low pixels are black
    imgeq = cv2.equalizeHist(img_masked)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgcl = clahe.apply(img)
    histeqimg.append(imgeq)
    claheimg.append(imgcl)

# Plotting layout: Top row = CLAHE and HistEq, Bottom center = Original
# Create figure and 2x6 GridSpec
# Create figure with custom layout
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 3, height_ratios=[1, 1])

# Row 1: CLAHE and Histogram Equalized
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(imgcl, cmap='gray')
ax1.set_title("CLAHE")
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 2])
ax2.imshow(imgeq, cmap='gray')
ax2.set_title("Histogram Equalized")
ax2.axis('off')

# Row 2: Centered original RGB image
ax3 = fig.add_subplot(gs[1, 1])
ax3.imshow(img_masked)
ax3.set_title("Original Image")
ax3.axis('off')

plt.tight_layout()
plt.show()



