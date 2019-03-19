import os,sys,math,time,io,argparse,json
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, utils, models
from PIL import Image



class PlainDataset(data.Dataset):
	def __init__(self, params):
		self.params = params
		# get input dirs
		if 'dirs' in params:
			self.input_dirs = params['dirs']
		else:
			self.input_dirs = [params['input_dir']]
			if 'gt_dir' in params: self.input_dirs.append(params['gt_dir'])
		# transform input dirs in list
		if not isinstance(self.input_dirs,list): self.input_dirs = [self.input_dirs]
		# check that all input dirs exist
		for cur_dir in self.input_dirs:
			if not os.path.isdir(cur_dir):
				raise ValueError('Directory {} not existing.'.format(cur_dir))
		# if list exists
		if 'list' in params and os.path.isfile(params['list']):
			# load files from there
			in_file = open(params['list'],"r")
			lines = in_file.read()
			in_file.close()
			# get filenames
			self.fns = [l for l in lines.split('\n')
			if l and os.path.isfile(os.path.join(self.input_dirs[0],l))]
		else:
			# load files from first folder
			self.fns = [l for l in os.listdir(self.input_dirs[0])
			if os.path.isfile(os.path.join(self.input_dirs[0],l))]
		# raise error if no images are found
		if len(self.fns)==0: raise ValueError('No images found.')


	def __getitem__(self, index):
		# get current filename
		cur_fn = self.fns[index]
		# load images
		images = [Image.open(os.path.join(d,cur_fn)).convert('RGB') for d in self.input_dirs]
		# convert in tensors
		images = [transforms.ToTensor()(img) for img in images]
		# resize if required
		if 'loadsize' in self.params:
			# get size
			sz = self.params['loadsize']
			# set it to be a list
			if not isinstance(sz,(list,tuple)): sz = [sz,sz]
			# define resize func
			resfunc = lambda x: F.interpolate(x.unsqueeze(0), size=(sz[0],sz[1]), mode='bilinear',align_corners=False).squeeze(0)
			# resize all images
			images = [resfunc(img) for img in images]
		# check if filenames must be returned
		if 'include_filenames' in self.params: images.append(cur_fn)
		# return images
		return images

	def __len__(self):
		return len(self.fns)
