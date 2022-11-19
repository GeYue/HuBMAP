import numpy as np
import pandas as pd

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import random
from glob import glob
import os, sys, shutil, time, copy, gc, cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, KFold
from typing import Counter
from augmentation import *

image_size = 1024  #672 #736 #768 #32*24

random_seed = 42

TRAIN = './data/png512/train'
MASKS = './data/png512/masks'
LABELS = './data/train.csv'

#------------------------------
def make_fold(fold=0):
	df = pd.read_csv('./data/train.csv')
	
	num_fold = 5

	# skf = KFold(n_splits=num_fold, shuffle=True, random_state=random_seed)
	
	# df.loc[:,'fold']=-1
	# for f,(t_idx, v_idx) in enumerate(skf.split(X=df['id'], y=df['organ'])):
	# 	df.iloc[v_idx,-1]=f

	# #check
	# if 0:
	# 	for f in range(num_fold):
	# 		train_df=df[df.fold!=f].reset_index(drop=True)
	# 		valid_df=df[df.fold==f].reset_index(drop=True)
			
	# 		print('fold %d'%f)
	# 		t = train_df.organ.value_counts().to_dict()
	# 		v = valid_df.organ.value_counts().to_dict()
	# 		for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
	# 			print('%32s %3d (%0.3f)  %3d (%0.3f)'%(k,t.get(k,0),t.get(k,0)/len(train_df),v.get(k,0),v.get(k,0)/len(valid_df)))
			
	# 		print('')
	# 		zz=0

	cv = StratifiedKFold(n_splits=num_fold, random_state=random_seed, shuffle=True)
	for fold_index, (_, val_index) in enumerate(cv.split(df, df["organ"])):
		df.loc[val_index, "fold"] = fold_index
	df = df.astype({"fold": 'int64'})

	# check
	print ("==========================================================================")
	for fold_index in range(num_fold):
		records = df[(df["fold"] == fold_index)]
		organ_counts = Counter(records["organ"].values)
		print(f"fold{fold_index}: {organ_counts}")
	print ("==========================================================================")
	
	train_df=df[df.fold!=fold].reset_index(drop=True)
	valid_df=df[df.fold==fold].reset_index(drop=True)
	return train_df,valid_df


def pad_to_multiple(image, mask, multiple=32, min_size=image_size):
	
	sh,sw,_ = image.shape
	ph = max(min_size,int(np.ceil(sh/32))*32) -sh
	pw = max(min_size,int(np.ceil(sw/32))*32) -sw
 
	image = np.pad(image, ((0,ph), (0,pw), (0,0)), 'constant', constant_values=0)
	mask  = np.pad(mask, ((0,ph), (0,pw)), 'constant', constant_values=0)
	return image, mask
	

####################################################################################################
import tifffile as tiff 

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


class HubmapDataset(Dataset):
	def __init__(self, df, augment=None, transforms=None):
		
		self.df = df
		self.augment = augment
		self.transforms = transforms
		self.length = len(self.df)

		ids = pd.read_csv(LABELS).id.astype(str).values
		#self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('_')[0] in ids]
		self.organ_to_label = {'kidney' : 0,
                               'prostate' : 1,
                               'largeintestine' : 2,
                               'spleen' : 3,
                               'lung' : 4}
	
	def __str__(self):
		string = ''
		string += '\tlen = %d\n' % len(self)
		
		d = self.df.organ.value_counts().to_dict()
		for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
			string +=  '%24s %3d (%0.3f) \n'%(k,d.get(k,0),d.get(k,0)/len(self.df))
		return string
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, index):
		#fname = self.fnames[index]
		d = self.df.iloc[index]
		organ = self.organ_to_label[d.organ]

		id = d['id']

		#image = cv2.cvtColor(cv2.imread(os.path.join(TRAIN,fname)), cv2.COLOR_BGR2RGB)
		#mask = cv2.imread(os.path.join(MASKS,fname),cv2.IMREAD_GRAYSCALE)

		#print (f"d.info---> {id} / {d.imgPath} ")
		image = tiff.imread(f"/mnt/ramdisk/train_images/{id}.tiff")
		mask = rle2mask(d["rle"], (image.shape[1], image.shape[0]))
		
		image = image.astype(np.float32)/255
		#mask  = mask.astype(np.float32)/255
		
		if self.transforms is not None:
			augdata = self.transforms(image=image, mask=mask)
			augimg, augmsk = augdata['image'], augdata['mask']
		#s = d.pixel_size/0.4 * (image_size/3000)
		#image = cv2.resize(image,dsize=None, fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
		#mask  = cv2.resize(mask, dsize=None, fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
		image = cv2.resize(image,dsize=(image_size,image_size),interpolation=cv2.INTER_LINEAR)
		mask  = cv2.resize(mask, dsize=(image_size,image_size),interpolation=cv2.INTER_LINEAR)
		
		if self.augment is not None:
			image, mask = self.augment(image, mask, organ)
		
		 
		r ={}
		r['index']= index
		r['id'] = id
		r['organ'] = torch.tensor([organ], dtype=torch.long)
		r['image'] = augimg #image_to_tensor(image)
		r['mask' ] = mask_to_tensor(augmsk) #mask_to_tensor(mask>0.5)
		return r

tensor_list = [
	'mask', 'image', 'organ',
]


def null_collate(batch):
	d = {}
	key = batch[0].keys()
	for k in key:
		v = [b[k] for b in batch]
		if k in tensor_list:
			v = torch.stack(v)
		d[k] = v
	
	d['mask'] = d['mask'].unsqueeze(1)
	d['organ'] = d['organ'].reshape(-1)
	return d

##############################################################################################################

def image_to_tensor(image, mode='bgr'): #image mode
	if mode=='bgr':
		image = image[:,:,::-1]
	x = image
	x = x.transpose(2,0,1)
	x = np.ascontiguousarray(x)
	x = torch.tensor(x, dtype=torch.float)
	return x

def tensor_to_image(x, mode='bgr'):
	image = x.data.cpu().numpy()
	image = image.transpose(1,2,0)
	if mode=='bgr':
		image = image[:,:,::-1]
	image = np.ascontiguousarray(image)
	image = image.astype(np.float32)
	return image

def mask_to_tensor(mask):
	x = mask
	x = torch.tensor(x, dtype=torch.float)
	return x

def tensor_to_mask(x):
	mask = x.data.cpu().numpy()
	mask = mask.astype(np.float32)
	return mask






########################################################################
def valid_augment5(image, mask, organ):
	#image, mask  = do_crop(image, mask, image_size, xy=(None,None))
	return image, mask


def train_augment5a(image, mask, organ):
	
	image, mask = do_random_flip(image, mask)
	image, mask = do_random_rot90(image, mask)
	
	for fn in np.random.choice([
		lambda image, mask : (image, mask),
		lambda image, mask : do_random_noise(image, mask, mag=0.1),
		lambda image, mask : do_random_contast(image, mask, mag=0.25),
		lambda image, mask : do_random_hsv(image, mask, mag=[0.30,0.30,0])
	],2): image, mask =  fn(image, mask)
 
	for fn in np.random.choice([
		lambda image, mask : (image, mask),
		lambda image, mask : do_random_rotate_scale(image, mask, angle=45,scale=[0.5,2]),
	],1): image, mask = fn(image, mask)
	
	return image, mask



def train_augment5b(image, mask, organ):
	image, mask = do_random_flip(image, mask)
	image, mask = do_random_rot90(image, mask)
	
	for fn in np.random.choice([
		lambda image, mask: (image, mask),
		lambda image, mask: do_random_noise(image, mask, mag=0.1),
		lambda image, mask: do_random_contast(image, mask, mag=0.40),
		lambda image, mask: do_random_hsv(image, mask, mag=[0.40, 0.40, 0])
	], 2): image, mask = fn(image, mask)
	
	for fn in np.random.choice([
		lambda image, mask: (image, mask),
		lambda image, mask: do_random_rotate_scale(image, mask, angle=45, scale=[0.50, 2.0]),
	], 1): image, mask = fn(image, mask)
	
	return image, mask

import albumentations as A
from albumentations.pytorch import ToTensorV2
data_transforms = {
    "train": A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
##          A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        A.CoarseDropout(max_holes=8, max_height=image_size//20, max_width=image_size//20,
                         min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ToTensorV2(transpose_mask=True),
        ], p=1.0),

    "valid": A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
        ToTensorV2(transpose_mask=True),
        ], p=1.0)
}


########################################################################

