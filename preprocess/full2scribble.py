
import cv2
import os
import numpy as np
import random
from skimage.morphology import skeletonize
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import label
import matplotlib.pyplot as plt

def full2scribble(full_label):
    height, width, _ = full_label.shape
    alpha_channel = full_label[:, :, 3]
    full_label_BW = np.zeros((height, width), np.uint8)
    full_label_BW[alpha_channel > 0] = 255

    def remove_corner(mask, coords):
        for coord in coords:
            x, y = coord
            mask[x-1][y-1] = 0
            mask[x-1][y] = 0
            mask[x-1][y+1] = 0
            mask[x][y-1] = 0
            mask[x][y] = 0
            mask[x][y+1] = 0
            mask[x+1][y-1] = 0
            mask[x+1][y] = 0
            mask[x+1][y+1] = 0
        return mask

    def scribblize(mask, ratio):
        sk = skeletonize(mask)
        i_mask = np.abs(mask - 1) // 255
        i_sk = skeletonize(i_mask)
        coords = corner_peaks(corner_harris(i_sk), min_distance=5)
        i_sk = remove_corner(i_sk, coords)
        
        label_sk = label(sk)
        n_sk = np.max(label_sk)
        n_remove = int(n_sk * (1-ratio))
        removes = random.sample(range(1, n_sk+1), n_remove)
        for i in removes:
            label_sk[label_sk == i] = 0
        sk = (label_sk > 0).astype('uint8')

        label_i_sk = label(i_sk)
        n_i_sk = np.max(label_i_sk)
        n_i_remove = n_i_sk - (n_sk - n_remove)
        print(f"n_sk: {n_sk}, n_remove: {n_remove}, n_i_sk: {n_i_sk}, n_i_remove: {n_i_remove}")

        removes = random.sample(range(1, n_i_sk+1), n_i_remove)
        for i in removes:
            label_i_sk[label_i_sk == i] = 0
        i_sk = (label_i_sk > 0).astype('uint8')
        return sk, i_sk

    labels = np.zeros((height, width), np.uint16)
    labels[full_label_BW > 0] = 1
    mask = (labels > 0).astype('uint8')
    sk, i_sk = scribblize(mask, ratio=1)
    scr = np.ones_like(mask) * 255
    scr[i_sk == 1] = 255 # foreground scribble 0
    scr[sk == 1] = 1 # background scribble 1

    return scr

if __name__ == '__main__':
    imageID = 0
    for root, dirs, files in os.walk("preprocess/mask"):
        for file in files:
            imageID = imageID + 1
            full_label_dir = "preprocess/mask/" + file
            scribble_label_dir = "preprocess/scribble/" + file
            full_label = cv2.imread(full_label_dir, cv2.IMREAD_UNCHANGED)
            scribble_label = full2scribble(full_label)
            cv2.imwrite(scribble_label_dir, scribble_label)
            print("Image " + str(imageID) + ": Finished!")