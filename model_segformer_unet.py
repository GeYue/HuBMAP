#coding=UTF-8

#from kaggle_hubmap_v2 import *
#from common import *

from mit import *


#################################################################

class RGB(nn.Module):
	IMAGE_RGB_MEAN = [0.485, 0.456, 0.406] #[0.5, 0.5, 0.5]
	IMAGE_RGB_STD  = [0.229, 0.224, 0.225] #[0.5, 0.5, 0.5]
	
	def __init__(self,):
		super(RGB, self).__init__()
		self.register_buffer('mean', torch.zeros(1,3,1,1))
		self.register_buffer('std', torch.ones(1,3,1,1))
		self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
		self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)
	
	def forward(self, x):
		x = (x-self.mean)/self.std
		return x


class Net(nn.Module):
	def load_pretrain(self,):
		checkpoint = cfg[self.arch]['checkpoint']
		print('load %s'%checkpoint)
		checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)  #True
		print(self.encoder.load_state_dict(checkpoint,strict=False))  #True
	
	
	def __init__(self, 
		submission=False, 
		arch='mit_b2', 
		extendLevel=True,
		decoder_cfg=dict(),
		):
		super(Net, self).__init__()
		decoder_dim = [512, 256, 128, 64, 32] #[256, 128, 64, 32, 16]

		if submission:
			self.output_type = ['inference']
			from smp_unet import UnetDecoder
		else:
			self.output_type = ['inference', 'loss']
			from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
		self.rgb = RGB()

		# ==== new level
		self.extendLevel = extendLevel
	
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

		self.arch = arch #'mit_b5'/ 'mit_b3' / 'mit_b2'
		self.encoder = cfg[self.arch]['builder']()
	
		encoder_dim = self.encoder.embed_dims
		#[64, 128, 320, 512]

		self.decoder = UnetDecoder(
			encoder_channels=[0, conv_dim] + encoder_dim,
			decoder_channels=decoder_dim,
			n_blocks=5,
			use_batchnorm=True,
			center=False,
			attention_type=None,
		)
		self.logit = nn.Sequential(
			nn.Conv2d(decoder_dim[-1], 1, kernel_size=1, padding=0),
		)
		self.aux = nn.ModuleList([
			nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
			#nn.Conv2d(decoder_dim, 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
		])
	
	
	def forward(self, batch):
		
		x = batch['image']
		x = self.rgb(x)
		
		B,C,H,W = x.shape
		encoder = self.encoder(x)
		#print([f.shape for f in encoder])
		
		conv = self.conv(x)
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

		logit = self.logit(last)
		#print(logit.shape)
		
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

def criterion_aux_loss(logit, mask):
	mask = F.interpolate(mask,size=logit.shape[-2:], mode='nearest')
	loss = F.binary_cross_entropy_with_logits(logit,mask)
	return loss

'''

x.shape
torch.Size([4, 3, 320, 320])


print([f.shape for f in feature])
[
torch.Size([4, 128, 80, 80]),
torch.Size([4, 256, 40, 40]),
torch.Size([4, 512, 20, 20]),
torch.Size([4, 1024, 10, 10])
]


'''
def run_check_net():
	batch_size = 2
	image_size = 768
	
	#---
	batch = {
		'image' : torch.from_numpy( np.random.uniform(-1,1,(batch_size,3,image_size,image_size)) ).float(),
		'mask'  : torch.from_numpy( np.random.choice(2,(batch_size,1,image_size,image_size)) ).float(),
		'organ' : torch.from_numpy( np.random.choice(5,(batch_size)) ).long(),
	}
	batch = {k:v.cuda() for k,v in batch.items()}
	
	
	
	net = Net().cuda()
	# torch.save({ 'state_dict': net.state_dict() },  'model.pth' )
	# exit(0)
	net.load_pretrain()
	
	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)
	
	print('batch')
	for k,v in batch.items():
		print('%32s :'%k, v.shape)
	
	print('output')
	for k,v in output.items():
		if 'loss' not in k:
			print('%32s :'%k, v.shape)
	for k,v in output.items():
		if 'loss' in k:
			print('%32s :'%k, v.item())

'''
class mit_b0(MixVisionTransformer):
	def __init__(self, **kwargs):
		super(mit_b0, self).__init__(
			patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
			qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
			drop_rate=0.0, drop_path_rate=0.1)


'''
def run_check_mit():
	batch_size = 4
	image_size = 768
	
	checkpoint = '/root/Downloads/mit_b0.pth'
	#---
	image = torch.from_numpy( np.random.uniform(-1,1,(batch_size,3,image_size,image_size)) ).float()
	image = image.cuda()
	
	net = mit_b0()
	
	state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
	net.load_state_dict(state_dict,strict=False)  #True
	#Unexpected key(s) in state_dict: "head.weight", "head.bias".
	
	net = net.cuda()
	feature = net(image)
	print([f.shape for f in feature])

# main #################################################################
if __name__ == '__main__':
	#run_check_mit()
	run_check_net()
