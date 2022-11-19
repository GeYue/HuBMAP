
import numpy as np
import pandas as pd

from glob import glob
import os, sys, shutil, time, copy, gc, cv2

TRAIN = './data/png512/train'
MASKS = './data/png512/masks'
LABELS = './data/train.csv'

df = pd.read_csv('./data/train.csv')

delIdx = df[df.id == 9791]
df = df.drop(delIdx.index)
delIdx = df[df.id == 22059]
df = df.drop(delIdx.index)

df.reset_index(drop=True)

imgs = glob("/mnt/ramdisk/png512/train/*.png")
msks = glob("/mnt/ramdisk/masks/*.png")

ndf = pd.DataFrame(columns=['id','imgPath', 'mskPath'])
idx = 0
for fname in imgs:
	file = fname.split('/')[-1]
	id = int(file.split('_')[0])
	imgPath = fname
	mskPath = fname.replace("train", "masks")
	ndf.loc[idx] = [id, imgPath, mskPath]
	idx += 1

newSet = df.merge(ndf, on=['id'], how='left')
print(newSet.shape)
newSet.to_csv("./newSet.csv")

