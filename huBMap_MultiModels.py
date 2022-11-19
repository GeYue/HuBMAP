#coding=UTF-8

import os
import cv2
import time
import random

import logging
import pandas as pd

import numpy as np
from itertools import repeat
import collections.abc

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import RandomSampler 
from torch.utils.data import SequentialSampler

import warnings
warnings.filterwarnings('ignore')

from dataset import *
from glob import glob

is_amp = False #False
root_dir = '.'
pretrain_dir = './model'

global_model = "segUnet" # "swinVx" / "segformer" / "pvtV2" / "pvtV2unet" / "segUnet"
model_arch = ""
extLevel = False #True

if global_model == "segformer" :
    from model_segformer import *
    model_arch = 'mit_b5'
elif global_model == "swinVx" :
    from model_swin_transformers import *
    model_arch = 'swin_small_patch4_window7_224_22k'
elif global_model == "pvtV2":
    from model_pvt_v2_daformer import *
    model_arch = 'pvt_v2_b4'
elif global_model == "pvtV2unet":
    from model_pvt_v2_smp_unet import *
    model_arch = 'pvt_v2_b5'
elif global_model == "segUnet":
    from model_segformer_unet import *
    model_arch = 'mit_b5'
else:
    print (f"Model type error!!!!")
    exit(-5)
sub_fold = global_model

start_lr   = 5e-5 #0.0001
batch_size = 1 #32 #32
early_stop_threshold = 120 # 30[for best_valid_early_stop] 150/200[for best_dice_early_stop]


import logging
logging.basicConfig(level=logging.INFO,
                    filename='run.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    #format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s :: %(message)s'
                    )
logger = logging.getLogger(__name__)
logger.info(f"\nlogger started. HuBMAP ---> 🔴🟡🟢 ")
logger.info(f"\tGlobal_model=={global_model}, arch=={model_arch}, random_seed=={random_seed}, extLevel=={extLevel}")
print (f"\tGlobal_model=={global_model}, arch=={model_arch}, random_seed=={random_seed}, extLevel=={extLevel}")

def message(mode='print', dice=False, valid=False):
	dicemark = validmark = '🐾'
	if mode==('print'):
		loss = batch_loss
	if mode==('log'):
		loss = train_loss
		#if (iteration % iter_save == 0): asterisk = '*'
		if dice: dicemark = '⚡️'
		if valid: validmark = '🥊'
	
	time_elapsed = time.time() - start_timer
	text = \
		('%0.2e   %08d %6.2f \t| '%(rate, iteration, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
		'%4.3f%s  %4.3f%s  %4.3f  %4.3f | '%(valid_loss[0], dicemark, valid_loss[1], validmark, valid_loss[2], valid_loss[3]) + \
		'%4.3f  %4.3f   | '%(*loss,) + \
		'%2.0fh  %02.0fm  %02.0fs' % ( time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60 )
	
	return text


def compute_dice_score(probability, mask):
    N = len(probability)
    p = probability.reshape(N,-1)
    t = mask.reshape(N,-1)

    p = p>0.5
    t = t>0.5
    uion = p.sum(-1) + t.sum(-1)
    overlap = (p*t).sum(-1)
    dice = 2*overlap/(uion+0.0001)
    return dice

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

CFG = {
    "lr": start_lr,
    "scheduler": 'CosineAnnealingLR',
    #"T_max": 'Defined in below code.'
    "min_lr": 0, #1e-6,
    "T_0": 25,
    "warmup_epochs": 0,
}

def fetch_scheduler(optimizer):
    if CFG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG['T_max'], 
                                                   eta_min=CFG['min_lr'])
    elif CFG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG['T_0'], 
                                                             eta_min=CFG['min_lr'])
    elif CFG['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG['min_lr'],)
    elif CFG['scheduler'] == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG['scheduler'] == None:
        return None
        
    return scheduler

def validate(net, valid_loader):
    valid_num = 0
    valid_probability = []
    valid_mask = []
    valid_loss = 0

    net = net.eval()
    start_timer = time.time()
    for t, batch in enumerate(valid_loader):

        net.output_type = ['loss', 'inference']
        with torch.no_grad():
            with amp.autocast(enabled = is_amp):

                batch_size = len(batch['index'])
                batch['image'] = batch['image'].cuda()
                batch['mask' ] = batch['mask' ].cuda()
                batch['organ'] = batch['organ'].cuda()

                output = net(batch)
                loss0  = output['bce_loss'].mean()

        valid_probability.append(output['probability'].data.cpu().numpy())
        valid_mask.append(batch['mask'].data.cpu().numpy())
        valid_num += batch_size
        valid_loss += batch_size*loss0.item()

        #debug
        if 0 :
            pass
            organ = batch['organ'].data.cpu().numpy()
            image = batch['image']
            mask  = batch['mask']
            probability  = output['probability']

            for b in range(batch_size):
                m = tensor_to_image(image[b])
                t = tensor_to_mask(mask[b,0])
                p = tensor_to_mask(probability[b,0])
                overlay = result_to_overlay(m, t, p )

                text = label_to_organ[organ[b]]
                draw_shadow_text(overlay,text,(5,15),0.7,(1,1,1),1)

                image_show_norm('overlay',overlay,min=0,max=1,resize=1)
                cv2.waitKey(0)

        print('\r %8d / %d  time:: %-100s '%(valid_num, len(valid_loader.dataset),(time.time() - start_timer)),end='',flush=True)

    assert(valid_num == len(valid_loader.dataset))
    print("\n")
    probability = np.concatenate(valid_probability)
    mask = np.concatenate(valid_mask)

    loss = valid_loss/valid_num

    dice = compute_dice_score(probability, mask)
    dice = dice.mean()

    torch.cuda.empty_cache()
    gc.collect()
    
    return [dice, loss,  0, 0]

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##################
## Main Entry 
##################
os.environ['TORCH_HOME'] = f"{pretrain_dir}"
if len(sys.argv) != 2:
        print(f"Parameters error!!!!\nIt should be: '{sys.argv[0]} 0[1/2/3/4...]'")
        sys.exit()
#elif sys.argv[1] not in ['d', 'r', 'b', 'bp', 'x', 'br' ,'xx']:
#        print(f"Parameters error!!!!\nIt should be: '{sys.argv[0]} d' / '{sys.argv[0]} r' to use 'DeBERTa' or 'RoBERTa' model.")
#        sys.exit()
else:
    fold_param = int(sys.argv[1])

fold = fold_param
logger.info(f"Current fold === {fold}")

out_dir = root_dir + '/output/' + sub_fold + '/fold-%d' % (fold)
initial_checkpoint = None #out_dir + '/checkpoint/fold00_00010920.bD-model-0.760.pth'

## setup  ----------------------------------------
for f in ['checkpoint','train','valid','backup'] : os.makedirs(out_dir +'/'+f, exist_ok=True)

    
logger.info('--- [START %s] %s' % (global_model, '-' * 64))


## dataset ----------------------------------------
logger.info('** dataset setting **')

train_df, valid_df = make_fold(fold)

train_dataset = HubmapDataset(train_df, transforms=data_transforms['train'])
valid_dataset = HubmapDataset(valid_df, transforms=data_transforms['valid'])

train_loader  = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset),
    batch_size  = batch_size,
    drop_last   = False,
    num_workers = 50,
    pin_memory  = True,
    worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    collate_fn = null_collate,
)

valid_loader = DataLoader(
    valid_dataset,
    sampler = SequentialSampler(valid_dataset),
    batch_size  = batch_size,
    drop_last   = False,
    num_workers = 50,
    pin_memory  = True,
    collate_fn = null_collate,
)


logger.info('fold = %s'%str(fold))
logger.info('train_dataset : \n%s\n'%(train_dataset))
logger.info('valid_dataset : \n%s\n'%(valid_dataset))


## net ----------------------------------------
logger.info('** net setting **')

scaler = amp.GradScaler(enabled = is_amp)
#decoder_cfg = {'decoder_dim': 320, }
net = Net(arch=model_arch, extendLevel=extLevel).cuda()

if initial_checkpoint is not None:
    f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
    start_iteration = f['iteration']
    start_epoch = f['epoch']
    state_dict  = f['state_dict']
    #net.load_state_dict(state_dict,strict=False)  #True

    ### only loading the "non-decoder" layers' parameters into model !!!!!!
    pd = {k: v for k, v in state_dict.items() if "decoder" not in k}
    md = net.state_dict()
    md.update(pd)
    net.load_state_dict(md)
    #start_iteration = start_epoch = 0

else:
    start_iteration = 0
    start_epoch = 0
    net.load_pretrain()

from dataset import image_size
logger.info('\tinitial_checkpoint = %s' % initial_checkpoint)
logger.info(f"\tLoaded ==> {cfg[net.arch]['checkpoint']}")

logger.info(f"\tAMP ==> {is_amp}")
print (f"\tAMP ==> {is_amp}")
logger.info(f"\tImage_size ===> {image_size}")
print(f"\tImage_size ===> {image_size}")


## optimiser ----------------------------------

def freeze_bn(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

if 0: ##freeze
    #for p in net.stem.parameters():   p.requires_grad = False
    for param in net.named_parameters():   
        if 'encoder' in param[0] or 'aux' in param[0]:
            param[1].requires_grad = False

    #freeze_bn(net)
    pass

#-----------------------------------------------

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)

logger.info('\toptimizer\n  %s'%(optimizer))

num_iteration = 450*len(train_loader) # 1000*len(train_loader)
iter_log   = len(train_loader)*3 #479
iter_valid = iter_log
#iter_save  = iter_log * 10

CFG['T_max'] = int(num_iteration/30)
scheduler = fetch_scheduler(optimizer)

#######################
## Start Training.....
#######################


logger.info('** start training here! **')
print ('\tbatch_size ==> %d '%(batch_size))
logger.info('   batch_size = %d '%(batch_size))
logger.info('                     \t\t|---------------VALID------------|---------------TRAIN/BATCH------------------')
logger.info('rate      iter       epoch\t| dice     loss     tp     tn    | loss           |  time           ')
logger.info('---------------------------------------------------------------------------------------------------------------')

valid_loss = np.zeros(4,np.float32)
train_loss = np.zeros(2,np.float32)
batch_loss = np.zeros_like(train_loss)
sum_train_loss = np.zeros_like(train_loss)
sum_train = 0

start_timer = time.time()
iteration = start_iteration
epoch = best_epoch = start_epoch
rate = 0

best_dice_score = 0
best_valid_loss = 10
show_dice_icon = show_valid_icon = False
while iteration <= num_iteration:
    for t, batch in enumerate(train_loader):

        """
        if iteration%iter_save==0:
            if iteration != start_iteration:
                torch.save({
                    'state_dict': net.state_dict(),
                    'iteration': iteration,
                    'epoch': epoch,
                }, out_dir + '/checkpoint/%08d.model.pth' %  (iteration))
                pass
        """

        if (iteration%iter_valid==0):
            valid_loss = validate(net, valid_loader)

            if valid_loss[0] > best_dice_score: 
                best_dice_score = valid_loss[0]
                best_epoch = epoch
                show_dice_icon = True
            if valid_loss[1] < best_valid_loss:
                best_valid_loss = valid_loss[1]
                best_valid_epoch = epoch
                show_valid_icon = True

            if show_valid_icon: pth_name = f"fold{fold:02}_{iteration:0>8d}.bV-model"
            if show_dice_icon : pth_name = f"fold{fold:02}_{iteration:0>8d}.bD-model"
            if show_dice_icon or show_valid_icon:
                if iteration != start_iteration:
                    if "bD-model" in pth_name:
                        removeFiles = glob(f"{out_dir}/checkpoint/*bD-model*.pth")
                    if "bV-model" in pth_name:
                        removeFiles = glob(f"{out_dir}/checkpoint/*bV-model*.pth")
                    for file in removeFiles: os.remove(file)
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%s-DS%.3f-VL%.3f.pth' %  (pth_name, valid_loss[0], valid_loss[1]))
                pass

        if (iteration%iter_log==0) or (iteration%iter_valid==0):
            print('\r', end='', flush=True)
            logger.info(message(mode='log', dice=show_dice_icon, valid=show_valid_icon))
            show_dice_icon = show_valid_icon = False

        ## early stop    
        if epoch - best_epoch >= early_stop_threshold:
        #if epoch - best_valid_epoch >= early_stop_threshold:
            print (f"Early stopping for no update in {early_stop_threshold} iterations!!!")
            logger.info(f"Early stopping for no update in {early_stop_threshold} iterations!!!")
            exit(100)

        # learning rate schduler ------------
        rate = get_learning_rate(optimizer)

        # one iteration update  -------------
        batch_size = len(batch['index'])
        batch['image'] = batch['image'].half().cuda()
        batch['mask' ] = batch['mask' ].half().cuda()
        batch['organ'] = batch['organ'].cuda()


        net.train()
        net.output_type = ['loss']
        if 1:
            with amp.autocast(enabled = is_amp):
                output = net(batch)
                loss0  = output['bce_loss'].mean()
                #loss1  = output['aux2_loss'].mean()

                loss1 = 0
                for i in range(len(net.aux)): 
                    if i == 0:  ## skip aux0 because it is so close to loss0, no need to add it into the loss calculation again.
                        pass
                    else:
                        loss1 += output['aux%d_loss'%i].mean()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            scaler.scale(loss0+0.2*loss1).backward()

            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()


        # print statistics  --------
        batch_loss[:2] = [loss0.item(),loss1.item()]
        sum_train_loss += batch_loss
        sum_train += 1
        if t % 100 == 0:
            train_loss = sum_train_loss / (sum_train + 1e-12)
            sum_train_loss[...] = 0
            sum_train = 0

        print('\r', end='', flush=True)
        print(message(mode='print'), end='', flush=True)
        epoch += 1 / len(train_loader)
        iteration += 1
        
    torch.cuda.empty_cache()
    gc.collect()
    


