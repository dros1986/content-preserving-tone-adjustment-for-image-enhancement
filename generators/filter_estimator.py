import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FilterEstimator(nn.Module):
	def __init__(self, params):
		super(FilterEstimator, self).__init__()
		self.params = params
		# define down and up block
		self.filter = self.get_filter(params['size'])

	def forward(self,x):
		return self.filter(x)

	def get_filter(self,sz):
		padsz = int((sz-1)/2)
		return nn.Sequential(
			nn.ReflectionPad2d(padsz),
			nn.Conv2d(3, 3, kernel_size=sz, stride=1, padding=0)
		)
