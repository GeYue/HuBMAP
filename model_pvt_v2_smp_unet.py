#from kaggle_hubmap_v3 import *
#from segmentation_lib.decoder.daformer import *

from pvt_v2 import *



#################################################################

class RGB(nn.Module):
	IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
	IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]
	
	def __init__(self, ):
		super(RGB, self).__init__()
		self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
		self.register_buffer('std', torch.ones(1, 3, 1, 1))
		self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
		self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)
	
	def forward(self, x):
		x = (x - self.mean) / self.std
		return x

class Net(nn.Module):
	
	def load_pretrain(self, ):
		pretain = pretrain_dir + '/' + self.arch + '.pth'
		print('load %s' % pretain)
		state_dict = torch.load(pretain, map_location=lambda storage, loc: storage)  # True
		print(self.encoder.load_state_dict(state_dict, strict=False))  # True
	
	def __init__(self,
	             #encoder=pvt_v2_b4,
	             decoder=None,
	             encoder_cfg=dict(),
	             decoder_cfg=dict(),
	             submission=False,
	             arch='pvt_v2_b4',
	             extendLevel=True,
	             ):
		super(Net, self).__init__()
		decoder_dim = [512, 256, 128, 64, 32] #[256, 128, 64, 32, 16]

		self.arch = arch
		self.extendLevel = extendLevel
		
		# ----
		if submission:
			self.output_type = ['inference']
			from smp_unet import UnetDecoder
		else:
			self.output_type = ['inference', 'loss']
			from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
		self.rgb = RGB()

		#https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
		conv_dim = 32
		self.conv = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, conv_dim, kernel_size=3, stride=1, padding=1, bias=False)
		)

		
		self.encoder = cfg[self.arch]['builder'](  #encoder(
			**encoder_cfg
		)
		encoder_dim = self.encoder.embed_dims
		# [64, 128, 320, 512]
		
		self.decoder = UnetDecoder(
			encoder_channels=[0, conv_dim] + encoder_dim,
			decoder_channels=decoder_dim,
			n_blocks=5,
			use_batchnorm=True,
			center=False,
			attention_type=None,
		)

		self.logit = nn.Sequential(
			nn.Conv2d(decoder_dim[-1], 1, kernel_size=1),
		)

		self.aux = nn.ModuleList([
			nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
		])

		# if self.extendLevel:
		# 	dim_sets = decoder_dim[::-1][1:]
		# 	self.aux = nn.ModuleList([
		# 		nn.Conv2d(dim_sets[i], 1, kernel_size=1, padding=0) for i in range(len(dim_sets))
		# 	])

	
	def forward(self, batch):
		
		x = batch['image']
		x = self.rgb(x)
		
		B, C, H, W = x.shape
		encoder = self.encoder(x)
		##print('encoder', [f.shape for f in encoder])

		conv = self.conv(x)
		##print('conv', conv.shape)

		# ---------------------------------
		if 1:
			feature = encoder[::-1]  # reverse channels to start from head of encoder
			head = feature[0]
			skip = feature[1:] + [conv, None]
			d = self.decoder.center(head)

			decoder = []
			for i, decoder_block in enumerate(self.decoder.blocks):
				# print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
				# print(decoder_block.conv1[0])
				# print('')
				s = skip[i]
				d = decoder_block(d, s)
				decoder.append(d)
			last = d
		#print('decoder',[f.shape for f in decoder])
		# ---------------------------------------------------------

		
		#---
		logit = self.logit(last)
		##print('logit',logit.shape)
		
		output = {}
		if 'loss' in self.output_type:
			output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, batch['mask'])
			#output['label_loss'] = criterion_binary_cross_entropy(logit, batch['mask'])
			
			for i in range(len(self.aux)):
				output['aux%d_loss' % i] = criterion_aux_loss(self.aux[i](encoder[i]), batch['mask'])
				#output['aux%d_loss' % i] = criterion_aux_loss(self.aux[i](decoder[len(self.aux)-i-1]), batch['mask'])
		
		if 'inference' in self.output_type:
			#probability_from_logit = torch.softmax(logit,1)
			probability_from_logit = torch.sigmoid(logit)
			output['probability_from_logit'] = probability_from_logit
			output['probability'] = probability_from_logit
		
		return output


def criterion_dice_loss(logit, mask):
	smooth = 1.
	
	batch_size, C, H, W = logit.shape
	p = torch.sigmoid(logit).reshape(batch_size, -1)
	t = mask.reshape(batch_size, -1)
	
	intersection = (p * t).sum(-1)
	union = p.sum(-1) + t.sum(-1)
	dice = 1 - (2 * intersection + smooth) / (union + smooth)
	dice = dice.mean()
	return dice


# http://jck.bio/pytorch_onehot/
def criterion_multi_binary_cross_entropy(logit, mask, organ):
	logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
	batch_size, C, H, W = logit.shape
	
	label = mask.long() * organ.reshape(batch_size, 1, 1, 1)
	onehot = torch.zeros((batch_size, num_organ + 1, H, W)).to(mask)
	onehot = onehot.scatter(1, label, 1)
	# onehot[:,0] = 1-onehot[:,0]
	
	loss = F.binary_cross_entropy_with_logits(logit, onehot)
	return loss

def criterion_binary_cross_entropy(logit, mask):
	logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
	loss = F.binary_cross_entropy_with_logits(logit, mask)
	return loss

def criterion_cross_entropy(logit, mask, organ):
	logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
	batch_size, C, H, W = logit.shape
	
	label = mask.long() * organ.reshape(batch_size, 1, 1, 1)
	
	#https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py
	log_softmax = F.log_softmax(logit,1)
	loss = F.nll_loss(log_softmax,label.squeeze(1))
	return loss



def criterion_aux_loss(logit, mask):
	mask = F.interpolate(mask, size=logit.shape[-2:], mode='nearest')
	loss = F.binary_cross_entropy_with_logits(logit, mask)
	return loss



def run_check_net():
	batch_size = 2
	image_size = 512 #800
	
	import numpy as np
	# ---
	batch = {
		'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
		'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float(),
		'organ': torch.from_numpy(np.random.choice(5, (batch_size, 1))).long(),
	}
	batch = {k: v.cuda() for k, v in batch.items()}
	
	net = Net().cuda()
	# print(net)
	# torch.save({ 'state_dict': net.state_dict() },  'model.pth' )
	# exit(0)
	net.load_pretrain()
	
	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)
	
	print('batch')
	for k, v in batch.items():
		print('%32s :' % k, v.shape)
	
	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print('%32s :' % k, v.shape)
	for k, v in output.items():
		if 'loss' in k:
			print('%32s :' % k, v.item())


# main #################################################################
if __name__ == '__main__':
	run_check_net()

