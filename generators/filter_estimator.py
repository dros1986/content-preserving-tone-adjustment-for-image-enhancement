import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.stats as st
import torch.nn.functional as F



class FilterEstimator(nn.Module):
	def __init__(self, params):
		super(FilterEstimator, self).__init__()
		self.params = params
		# define filter type
		self.ftype = self.params['filter_type'] if 'filter_type' in self.params else 'rgb'
		# define filter parameters
		if self.ftype == 'rgb':
			# create 4D filter
			kernel = self.gaussian_filter()
			new_f = torch.zeros(3,3,kernel.size(0),kernel.size(1))
			for i in range(3):
				new_f[i,i,:,:] = kernel
			# save parameter
			self.filter = torch.nn.Parameter(new_f)
		else:
			self.filter = torch.nn.Parameter(self.gaussian_filter())
		# create padding layer
		self.pad = nn.ReflectionPad2d(int((self.params['size']-1)/2))


	def forward(self,x):
		# pad input
		x = self.pad(x)
		# convolve with filter
		if self.ftype == 'rgb':
			x = F.conv2d(x, self.filter)
		else:
			new_f = torch.zeros(3,3,self.filter.size(0),self.filter.size(1)).to(x.device)
			for i in range(3): new_f[i,i,:,:] = self.filter
			x = F.conv2d(x, new_f)
		# return image
		return x


	def gaussian_filter(self):
		# create a 2D gaussian kernel array
		kernlen = self.params['size']
		nsig = self.params['sigma']
		interval = (2*nsig+1.)/(kernlen)
		x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
		kern1d = np.diff(st.norm.cdf(x))
		kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
		kernel = kernel_raw/kernel_raw.sum()
		# convert to torch
		kernel = torch.from_numpy(kernel)
		return kernel
