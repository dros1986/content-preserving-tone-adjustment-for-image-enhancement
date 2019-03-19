import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Block(nn.Module):
	def __init__(self, inner_ch, outer_ch, skip, down_leak, up_leak, inner):
		'''
		@param nemb size of the embedding
		@param nch number of input channels
		@param powf 2^nch is the number of filters in LAST layer before projection
		@param insz size of the input tensor
		@param minsz minimum size of the map before calculating embedding
		'''
		super(Block, self).__init__()
		# define down and up block
		self.db = self.down_block(outer_ch,inner_ch, down_leak)
		self.ub = self.up_block(inner_ch,outer_ch, up_leak)
		self.skip = skip
		self.down_leak = down_leak
		self.up_leak = up_leak
		self.inner = inner

	def forward(self,x):
		if self.inner:
			inmap = self.db(x)
			core = self.inner(inmap)
			if self.skip:
				if not inmap.size()==core.size():
					core = F.interpolate(core, size=(inmap.size(2),inmap.size(3)), mode='nearest')
				outmap = self.ub(core+inmap)
			else:
				outmap = self.ub(core)
		else:
			outmap = self.ub(self.db(x))
		return outmap

	def down_block(self,inch,outch,down_leak):
		return nn.Sequential(
			nn.Conv2d(inch, outch, kernel_size=4, stride=2, padding=1),
			nn.PReLU(num_parameters=outch, init=0.25),
		)

	def up_block(self,inch,outch,up_leak):
		return nn.Sequential(
			nn.Conv2d(inch, 4*outch, kernel_size=3, stride=1, padding=1),
			nn.PReLU(num_parameters=4*outch, init=0.25),
			nn.PixelShuffle(upscale_factor=2),
		)

class Autoencoder(nn.Module):
	def __init__(self, params):
		super(Autoencoder, self).__init__()
		# get model parameters
		self.params = params
		# save attributes
		self.nch = 3
		self.powf = params['powf']
		self.insz = params['insz']
		self.minsz = params['minsz']
		skip = params['skip']
		down_leak = params['down_leak']
		up_leak = params['up_leak']
		# calculate number of blocks to arrive to minsz x minsz
		nblocks = int(math.log(float(self.insz)/float(self.minsz), 2))
		self.nblocks = nblocks
		# define outer/inner
		nf = []
		for i in range(nblocks):
			# outer_nf = self.nch if i==0 else 2**(i+self.powf)
			outer_nf = 2**(i+self.powf)
			inner_nf = 2**(i+self.powf+1)
			nf.append((inner_nf,outer_nf))
		# define network
		self.ci = nn.Conv2d(3, 2**self.powf, kernel_size=3, stride=1, padding=1)
		self.net = self.create_net(nf, skip, down_leak, up_leak)
		self.cf = nn.Conv2d(2**self.powf, 3, kernel_size=3, stride=1, padding=1)

	def create_net(self,nf,skip,down_leak,up_leak):
		if len(nf)>0:
			return Block(nf[0][0],nf[0][1],skip,down_leak,up_leak,
						self.create_net(nf[1:],skip,down_leak,up_leak))
		else:
			return None

	def forward(self,x):
		x = F.relu(self.ci(x))
		x = self.net(x)
		x = self.cf(x)
		return x


if __name__ == '__main__':
	# define net
	params = {
		'generator':{
			'powf':2,
			'insz':256,
			'minsz':8,
			'skip':True
		}
	}
	# define network
	netG = Autoencoder(params)
	netG.cuda()
	print(netG)
	# define input
	inp = torch.rand(2,3,256,256).cuda()
	# calculate output
	# import ipdb; ipdb.set_trace()
	out = netG(inp)
	print(out.size())
