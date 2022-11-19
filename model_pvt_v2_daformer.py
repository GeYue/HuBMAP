#from kaggle_hubmap_kv3 import *
from daformer import *
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

from dataset import image_size
class Net(nn.Module):
	
	def load_pretrain(self,):
		checkpoint = cfg[self.arch]['checkpoint']
		print('load %s'%checkpoint)
		checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)  #True
		print(self.encoder.load_state_dict(checkpoint,strict=False))  #True
 
	def __init__(self,
	             #encoder=pvt_v2_b4,
	             decoder=daformer_conv3x3,
	             encoder_cfg=dict(img_size=image_size),
	             decoder_cfg=dict(),
	             submission=False,
	             arch='pvt_v2_b4',
	             extendLevel=False,
	             ):
		super(Net, self).__init__()
		if submission:
			self.output_type = ['inference']
		else:
			self.output_type = ['inference', 'loss']
		decoder_dim = decoder_cfg.get('decoder_dim', 320)
		
		# ----
		self.arch = arch #'pvt_v2_b4'
		self.rgb = RGB()

		self.extendLevel = extendLevel
		if extendLevel:
			# ==== new level
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
		if extendLevel:
			encoder_dim=[conv_dim] + encoder_dim
		else:
			pass
		self.decoder = decoder(
			encoder_dim=encoder_dim,
			decoder_dim=decoder_dim,
		)
		self.logit = nn.Sequential(
			nn.Conv2d(decoder_dim, 1, kernel_size=1),
		)
		self.aux = nn.ModuleList([
			nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
			#nn.Conv2d(decoder_dim, 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
		])
	
	def forward(self, batch):
		
		x = batch['image']
		x = self.rgb(x)
		
		B, C, H, W = x.shape
		encoder = self.encoder(x)
		#print([f.shape for f in encoder])
		
		if self.extendLevel:
			conv = self.conv(x)
			#print('conv', conv.shape)
			last, decoder = self.decoder([conv] + encoder)
			scale_factor = 2
		else:
			last, decoder = self.decoder(encoder)
			scale_factor = 4
		logit = self.logit(last)
		#print(logit.shape)
		#print([f.shape for f in decoder])
		logit = F.interpolate(logit, size=None, scale_factor=scale_factor, mode='bilinear', align_corners=False) ## scale_factor=4 when w/o new layer self.conv
		
		output = {}
		if 'loss' in self.output_type:
			output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['mask'])
			for i in range(len(self.encoder.embed_dims)):
				output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](encoder[i]),batch['mask'])
			#for i in range(len(self.aux)):
			#	output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](decoder[i]),batch['mask'])
			
		if 'inference' in self.output_type:
			output['probability'] = torch.sigmoid(logit)

		#probability_from_logit = torch.sigmoid(logit)
		#output['probability'] = probability_from_logit
		
		return output
 
def criterion_aux_loss(logit, mask):
	mask = F.interpolate(mask,size=logit.shape[-2:], mode='nearest')
	loss = F.binary_cross_entropy_with_logits(logit,mask)
	return loss 


def run_check_net():
	batch_size = 2
	#image_size = image_size #800
	
	# ---
	batch = {
		'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
		'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float(),
		'organ': torch.from_numpy(np.random.choice(5, (batch_size, 1))).long(),
	}
	batch = {k: v.cuda() for k, v in batch.items()}
	
	net = Net().cuda()
	
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
	# run_check_mit()
	run_check_net()
