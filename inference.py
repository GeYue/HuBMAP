
import sys, os 
#sys.path.append('../input/hubmap-submit-06') 
#sys.path.append('../input/hubmap-submit-06/[third_party]')   

import importlib

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

import pandas as pd
import numpy as np
import cv2

import tifffile as tiff
from timeit import default_timer as timer

def read_tiff(image_file, mode='rgb'):
	image = tiff.imread(image_file)
	image = image.squeeze()
	if image.shape[0] == 3:
		image = image.transpose(1, 2, 0)
	if mode=='bgr':
		image = image[:,:,::-1]
	image = np.ascontiguousarray(image)
	return image

print('import ok')



#-- configure ---------------------------------------------
image_size = 768 #512

organ_threshold = {
    'Hubmap': {
        'kidney'        : 0.40,
        'prostate'      : 0.40,
        'largeintestine': 0.40,
        'spleen'        : 0.40,
        'lung'          : 0.10,
    },
    'HPA': {
        'kidney'        : 0.50,
        'prostate'      : 0.50,
        'largeintestine': 0.50,
        'spleen'        : 0.50,
        'lung'          : 0.10,
    },
}

data_source =['Hubmap', 'HPA']
organ = ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']


#data_source =['Hubmap',]
#organ = ['spleen']


submit_type  = 'local-cv'    
#submit_type  = 'local-test'    
#submit_type  = 'kaggle'   


#--- helper ----------
def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t)/60
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min)
	
	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)
	
	else:
		raise NotImplementedError

class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

## model ##################################
#from coat import *
#from daformer import *

model = [
    dotdict(
        is_use = 0,
        module = 'model_pvt_v2_daformer', #'model_daformer_coat',
        desc = 'Net:: encoder --> PVT2B4 / decoder --> DaFormer',
        param= {'submission': True, 'arch': 'pvt_v2_b4', 'extendLevel': True}, #{'encoder': coat_lite_medium, 'decoder':daformer_conv3x3},
        checkpoint = [
            #'../input/hubmap-submit-06-weight0/daformer_conv3x3-coat_lite_medium-aug5b-768-fold-3-swa.pth'
            './kaggle/upload_finetuned/fold00_00013020.bD-model-DS0.818-VL0.094.pth',
            './kaggle/upload_finetuned/fold01_00030879.bD-model-DS0.805-VL0.127.pth',
            './kaggle/upload_finetuned/fold02_00030879.bD-model-DS0.824-VL0.087.pth',
            './kaggle/upload_finetuned/fold03_00020727.bD-model-DS0.819-VL0.084.pth',
            './kaggle/upload_finetuned/fold04_00029187.bD-model-DS0.690-VL0.111.pth',

        ],
    ),

    dotdict(
        is_use = 1,
        module = 'model_segformer', #'model_daformer_coat',
        desc = 'Net:: encoder --> SegFormer_mit-bX / decoder --> SegformerDecoder',
        param= {'submission': True, 'arch': 'mit_b3', 'extendLevel': False}, #{'encoder': coat_lite_medium, 'decoder':daformer_conv3x3},
        checkpoint = [
            #'../input/hubmap-submit-06-weight0/daformer_conv3x3-coat_lite_medium-aug5b-768-fold-3-swa.pth'
            './test/fold00_00015015.bD-model-0.732',
            './test/fold01_00010800.bD-model-0.820',
            './test/fold02_00025560.bD-model-0.802',
            './test/fold03_00004644.bD-model-0.769',
            './test/fold04_00023643.bD-model-0.803.pth',

        ],
    ),

    dotdict(
        is_use = 1,
        module = 'model_pvt_v2_smp_unet', #'model_daformer_coat',
        desc = 'Net:: encoder --> pvtV2_b5 / decoder --> UNetDecoder',
        param= {'submission': True, 'arch': 'pvt_v2_b5', 'extendLevel': True}, #{'encoder': coat_lite_medium, 'decoder':daformer_conv3x3},
        checkpoint = [
            #'../input/hubmap-submit-06-weight0/daformer_conv3x3-coat_lite_medium-aug5b-768-fold-3-swa.pth'
            './output/pvtV2unet/fold-0/checkpoint/fold00_00010500.bD-model-DS0.775-VL0.076.pth',
            #'./output/pvtV2unet/fold-1/checkpoint/fold01_00023604.bV-model-0.799.pth',

        ],
    ),
]


## dataset ##################################

if submit_type == 'local-cv':
    valid_file = './data/valid_df.fold3.csv' #'../input/hubmap-submit-06/valid_df.fold3.csv'
    tiff_dir   = './data/train_images'

if (submit_type == 'local-test') or (submit_type == 'kaggle'):
    valid_file = '../input/hubmap-organ-segmentation/test.csv'
    tiff_dir   = '../input/hubmap-organ-segmentation/test_images'

valid_df = pd.read_csv(valid_file)
valid_df.loc[:,'img_area']=valid_df['img_height']*valid_df['img_width']#sort by biggest image first for memory debug
valid_df = valid_df.sort_values('img_area').reset_index(drop=True)
print('load valid_df ok')


def image_to_tensor(image, mode='rgb'):
    if  mode=='bgr' :
        image = image[:,:,::-1]
    
    x = image.transpose(2,0,1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x)
    return x

# # --- rle ---------------------------------
def rle_decode(rle, height, width , fill=255, dtype=np.uint8):
	s = rle.split()
	start, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	start -= 1
	mask = np.zeros(height*width, dtype=dtype)
	for i, l in zip(start, length):
		mask[i:i+l] = fill
	mask = mask.reshape(width,height).T
	mask = np.ascontiguousarray(mask)
	return mask


def rle_encode(mask):
	m = mask.T.flatten()
	m = np.concatenate([[0], m, [0]])
	run = np.where(m[1:] != m[:-1])[0] + 1
	run[1::2] -= run[::2]
	rle =  ' '.join(str(r) for r in run)
	return rle


## submission ##################################

def do_local_validation():
    print('\tlocal validation ...')
    
    submit_df = pd.read_csv('submission.csv').fillna('')
    submit_df = submit_df.sort_values('id')
    truth_df  = valid_df.sort_values('id')
    
    lb_score = []
    num = len(submit_df)
    for i in range(num):
        t_df = truth_df.iloc[i]
        p_df = submit_df.iloc[i]
        t = rle_decode(t_df.rle, t_df.img_height, t_df.img_width, 1)
        p = rle_decode(p_df.rle, t_df.img_height, t_df.img_width, 1)
        
        dice = 2*(t*p).sum()/(p.sum()+t.sum())
        lb_score.append(dice)
        
        if 0:
            overlay = result_to_overlay(p, t)
            image_show_norm('overlay', overlay, min=0, max=1, resize=0.10)
            cv2.waitKey(1)

    truth_df.loc[:,'lb_score']=lb_score
    for organ in ['all', 'kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
        if organ != 'all':
            d = truth_df[truth_df.organ == organ]
        else:
            d = truth_df
        print('\t%f\t%-20s\t%f' % (len(d) / len(truth_df), organ, d.lb_score.mean()))
        
    
def load_net(model):
    print('\tload %s ... '%(model.desc),end='',flush=True)
    M = importlib.import_module(model.module)
    num = len(model.checkpoint)
    net = []
    for f in range(num):
        n = M.Net(**model.param)
        n.load_state_dict(
            torch.load(model.checkpoint[f], map_location=lambda storage, loc: storage) ['state_dict'],
            strict=False)
        n.cuda()
        n.eval()
        net.append(n)
        
    print('ok!')
    return net


def do_tta_batch(image, organ):
    
    batch = { #<todo> multiscale????
        'image': torch.stack([
            image,
            torch.flip(image,dims=[1]),
            torch.flip(image,dims=[2]),
        ]),
        'organ': torch.Tensor(
            [[organ_to_label[organ]]]*3
        ).long()
    }
    return batch

def undo_tta_batch(probability):
    probability[0] = probability[0]
    probability[1] = torch.flip(probability[1],dims=[1])
    probability[2] = torch.flip(probability[2],dims=[2])
    probability = probability.mean(0, keepdims=True)
    probability = probability[0,0].float()
    return probability

def do_submit(): 
    print('** submit_type  = %s *******************'%submit_type)

    all_net = [ load_net(m) for m in model if m.is_use==1 ]
    
    result = []
    start_timer = timer()
    for i,d in valid_df.iterrows():
        id = d['id']
        if (d['data_source'] in data_source) and (d['organ'] in organ):
            
            tiff_file = tiff_dir +'/%d.tiff'%id
            tiff = read_tiff(tiff_file, 'rgb') 
            tiff = tiff.astype(np.float32)/255
            H,W,_ = tiff.shape
            
            if 0:
                s = d.pixel_size/0.4 * (image_size/3000)
                h = int(np.ceil(int(H*s)/32)*32)
                w = int(np.ceil(int(W*s)/32)*32) 
                image = cv2.resize(tiff,dsize=(w,h),interpolation=cv2.INTER_LINEAR)
            else: 
                #or just resize to h,w = 768
                if tiff.shape[0] > image_size:
                    image = cv2.resize(tiff,dsize=(image_size,image_size),interpolation=cv2.INTER_AREA)
                else:
                    image = cv2.resize(tiff,dsize=(image_size,image_size),interpolation=cv2.INTER_CUBIC)
            
            image = image_to_tensor(image, 'rgb')
            batch = { k:v.cuda() for k,v in do_tta_batch(image, d.organ).items() }
    
            use = 0
            probability = 0
            with torch.no_grad():
                with amp.autocast(enabled = False):
                    
                    for net in all_net:
                        for n in net:
                            use += 1
                            output = n(batch)#data_parallel(net, batch) #
                            probability += \
                                F.interpolate(output['probability'], size=(d.img_height,d.img_width),
                                              mode='bilinear',align_corners=False, antialias=True )
                       
                    probability = undo_tta_batch(probability/use)
            #---
            probability = probability.data.cpu().numpy()
            p = probability>organ_threshold[d.data_source][d.organ] 
            rle = rle_encode(p)
        else:
            rle = ''
        
        #----
        if 0: #debug
            image = cv2.cvtColor(tiff, 4).astype(np.float32)/255 #cv2.COLOR_RGB2BGR=4
            mask  = rle_decode(d.rle, d.img_height, d.img_width, 1) #None
            overlay = result_to_overlay(image, mask, probability)
            
            #image_show('image',image, resize=0.25)
            image_show('overlay',overlay, resize=0.25)
            cv2.waitKey(0)
            pass
        
        result.append({ 'id':id, 'rle':rle, })
        print('\r', '\tsubmit ... %3d/%3d %s'%(i, len(valid_df), time_to_str(timer() - start_timer,'sec')), end='',flush=True)
    print('\n')
    
    #---
    submit_df = pd.DataFrame(result)
    submit_df.to_csv('submission.csv',index=False)
    print(submit_df)
    print('\tsubmit_df ok!')
    print('')
    
    if submit_type == 'local-cv':
        do_local_validation()
        
    if submit_type == 'local-test':
        import matplotlib.pyplot as plt 
        m = tiff
        p = probability
        
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 3, 1); plt.imshow(m); plt.axis('OFF'); plt.title('image')
        plt.subplot(1, 3, 2); plt.imshow(p*255); plt.axis('OFF'); plt.title('mask')
        plt.subplot(1, 3, 3); plt.imshow(m); plt.imshow(p*255, alpha=0.4); plt.axis('OFF'); plt.title('overlay')
        plt.tight_layout()
        plt.show()

##--------------------------------------------------------------------------------------
organ_meta = dotdict(
	kidney = dotdict(
		label = 0, #1,
		um    = 0.5000,
		ftu   ='glomeruli',
	),
	prostate = dotdict(
		label = 1, #2,
		um    = 6.2630,
		ftu   ='glandular acinus',
	),
	largeintestine = dotdict(
		label = 2, #3,
		um    = 0.2290,
		ftu   ='crypt',
	),
	spleen = dotdict(
		label = 3, #4,
		um    = 0.4945,
		ftu   ='white pulp',
	),
	lung = dotdict(
		label = 4, #5,
		um    = 0.7562,
		ftu   ='alveolus',
	),
)

organ_to_label = {k: organ_meta[k].label for k in organ_meta.keys()}
label_to_organ = {v:k for k,v in organ_to_label.items()}

do_submit()
