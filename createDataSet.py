
import numpy as np
import pandas as pd

import glob
import os, sys, shutil, time, copy, gc, cv2
import tifffile as tiff 
from tqdm import tqdm

# https://www.kaggle.com/paulorzp/rle-functions-run-length-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


df = pd.read_csv('./data/train.csv')
image_ids = df.id
image_files = glob.glob(f"./data/train_images/*")

for id, img in tqdm(zip(image_ids, image_files), total = len(image_ids)):

    img = tiff.imread(img)
    mask = rle2mask(df[df["id"]==id]["rle"].iloc[-1], (img.shape[1], img.shape[0]))
    
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(mask, cmap='coolwarm', alpha=0.5)
    plt.savefig("./image.jpg")
    plt.close()

    table.add_data(
        id, 
        wandb.Image(img), 
        wandb.Image(mask),
        wandb.Image(cv2.cvtColor(cv2.imread("./image.jpg"), cv2.COLOR_BGR2RGB))
    )
