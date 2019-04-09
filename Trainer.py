import os,sys,math,time,io,argparse,json,traceback,collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, utils, models
from tensorboardX import SummaryWriter
from multiprocessing import cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
# losses and generators
from losses import get_loss
from generators import get_generator, get_generator_name
from datasets import get_dataset


class cols:
	GREEN = '\033[92m'; BLUE = '\033[94m'; CYAN = '\033[36m';
	LIGHT_GRAY = '\033[37m'; ENDC = '\033[0m'


class Trainer(nn.Module):
	def __init__(self, params):
		super(Trainer, self).__init__()
		# save params
		self.params = params
		# set device
		self.device = torch.device(params['training']['device'])
		# set niter to -1
		self.niter = -1
		# set attribute for best score
		self.best_psnr = None
		# create generator
		self.netG = get_generator(self.params['generator'])
		print(self.netG)
		# move it to device
		self.netG.to(self.device)
		# define output filename
		self.name = get_generator_name(self.params['generator'])
		# define dirs
		self.base_dir = os.path.join('./checkpoints', self.params['exp_name'])
		self.model_dir = self.base_dir
		self.logs_dir = self.base_dir
		self.images_dir = self.base_dir
		self.out_dir = os.path.join(self.base_dir,'regen')
		# create them
		if not os.path.isdir(self.base_dir): os.makedirs(self.base_dir)
		if not os.path.isdir(self.model_dir): os.makedirs(self.model_dir)
		if not os.path.isdir(self.logs_dir): os.makedirs(self.logs_dir)
		if not os.path.isdir(self.images_dir): os.makedirs(self.images_dir)
		if not os.path.isdir(self.out_dir): os.makedirs(self.out_dir)
		# if not training, do not continue
		if not self.training: return
		# get loss
		self.loss = get_loss(self.params['training']['loss'])
		# create generator optimizer
		self.optimG = torch.optim.Adam(	self.netG.parameters(),
										lr=self.params['training']['lr'],
										weight_decay=self.params['training']['weight_decay'])
		# init weights
		for m in self.modules():
				if isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)


	def inf_gen(self, data_loader):
		while True:
			for inp,gt in data_loader:
				yield inp,gt

	def create_dataloaders(self):
		# create training dataloader
		self.training_data_loader = data.DataLoader(
				get_dataset(self.params['dataset'],phase='train'),
				batch_size = self.params['dataset']['params']['train']['batch_size'],
				shuffle = True,
				num_workers = self.params['dataset']['params']['train']['num_workers'],
				drop_last = True
		)
		# create validation dataloader
		self.validation_data_loader = data.DataLoader(
				get_dataset(self.params['dataset'],phase='val'),
				batch_size = self.params['dataset']['params']['val']['batch_size'],
				shuffle = False,
				num_workers = self.params['dataset']['params']['val']['num_workers'],
				drop_last = False
		)
		# create infinite data gen
		self.train_datagen = self.inf_gen(self.training_data_loader)


	def save(self,best=False):
		# set basename
		# basename = os.path.join(self.model_dir, self.name)
		basename = os.path.join(self.model_dir, 'model')
		# append _best if is best
		if best: basename += '_best'
		# create state
		state = {  'netG' : self.netG.state_dict(),
				 'optimG' : self.optimG.state_dict(),
				  'niter' : self.niter,
				   'psnr' : self.best_psnr }
		# check if loss has state
		if callable(getattr(self.loss, "get_state", None)):
			state['loss'] = self.loss.get_state()
		# save model
		torch.save(state, basename+'.pth')
		# save json
		with open(basename+'.json', 'w') as outfile:
			json.dump(self.params, outfile, indent=4, sort_keys=False)


	def load(self,best=False):
		# define filename
		# basename = os.path.join(self.model_dir, self.name)
		basename = os.path.join(self.model_dir, 'model')
		if best:
			fn = basename + '_best.pth'
		else:
			fn = basename + '.pth'
		# if file exists
		if os.path.isfile(fn):
			# load state
			state = torch.load(fn)
			# load weights
			self.netG.load_state_dict(state['netG'])
			self.optimG.load_state_dict(state['optimG'])
			self.niter = state['niter']
			self.best_psnr = state['psnr'] if 'psnr' in state else None
			# check if loss has state
			if callable(getattr(self.loss, "get_state", None)):
				self.loss.set_state(state['loss'])
			# warn that weights have been loaded
			print('Parameters loaded from file {}'.format(fn))

	def forward(self, x):
		return self.netG(x)

	def trainG(self):
		# reset grads
		self.netG.zero_grad()
		self.optimG.zero_grad()
		# get new data
		inp,gt = self.train_datagen.__next__()
		# move to device
		inp = inp.to(self.device)
		gt = gt.to(self.device)
		# regenerate image
		regen = self.netG(inp)
		# calculate loss
		l = self.loss(regen,gt)
		# backward
		l.backward()
		# update weights
		self.optimG.step()
		# return
		return l.item(), inp, gt, regen


	def train(self):
		# create dataloaders
		self.create_dataloaders()
		# load if required
		self.load()
		# create tensorboardX writer
		writer = SummaryWriter(self.logs_dir)
		# for each iteration
		for self.niter in range(self.niter+1, self.params['training']['niters']):
			try:
				# get starting time
				start_time = time.time()
				# train generator
				g_cost, inp, gt, regen = self.trainG()
				# view losses on tensorboard
				writer.add_scalar('G_cost', g_cost, self.niter)
				# save images if required
				if self.niter % self.params['training']['show_images_every'] == 0:
					inp = torch.clamp(inp,0,1)
					gt = torch.clamp(gt,0,1)
					regen = torch.clamp(regen,0,1)
					nrow = int(math.sqrt(inp.size(0)))
					# basename = os.path.join(self.images_dir, self.name)
					utils.save_image(  inp, os.path.join(self.images_dir,'img_in.png'),nrow=nrow)
					utils.save_image(   gt, os.path.join(self.images_dir,'img_gt.png'),nrow=nrow)
					utils.save_image(regen, os.path.join(self.images_dir,'img_out.png'),nrow=nrow)
				# save model
				if self.niter > 0 and self.niter % self.params['training']['save_model_every'] == 0:
					self.save()
				# validate
				if self.niter > 0 and self.niter % self.params['training']['evaluate_every'] == 0:
					cur_psnr = self.validate()
					writer.add_scalar('psnr', cur_psnr, self.niter)

				# get time
				elapsed_time = time.time() - start_time
				if self.niter % 20 == 0:
					# define string
					s = \
						( \
						 cols.BLUE + '[{:07d}/{:07d}]' + \
						 cols.CYAN  + ' tm: ' + cols.BLUE + '{:.4f}' + \
						 cols.LIGHT_GRAY + ' G_cost: ' + cols.GREEN + '{:.4f}' + cols.ENDC \
						).format(self.niter, self.params['training']['niters'], elapsed_time, g_cost)
					# print it
					print(s)
			except OSError as e:
				print(e)
				self.train_datagen = self.inf_gen(self.training_data_loader)
			except StopIteration:
				self.train_datagen = self.inf_gen(self.training_data_loader)
			except KeyboardInterrupt:
				print('Quitting training.')
				sys.exit()
			except RuntimeError as re:
				print(re)
				sys.exit()
			except Exception as e:
				print(traceback.format_exc())
				self.train_datagen = self.inf_gen(self.training_data_loader)


	def validate(self):
		# set gen in test mode
		self.netG.eval()
		# set best
		vals = None
		# for each val batch
		for inp,gt in tqdm(self.validation_data_loader):
			# move to device
			inp, gt = inp.to(self.device), gt.to(self.device)
			# regen
			with torch.no_grad():
				out = self.netG(inp)
			# compare
			mse = torch.pow(out-gt,2).view(out.size(0),-1).mean(1)
			# mse = torch.pow(out-gt,2).mean(-1).mean(-1).mean(-1)
			# sobstitute where mse in 0 (to avoid inf)
			# mse = torch.where(mse==0, torch.tensor([1e-10]).to(mse.device), mse)
			mse = torch.max(mse,torch.tensor([1e-6]).to(mse.device))
			# calculate current psnr
			cur_vals = 10*torch.log10(1./mse)
			# append
			vals = cur_vals if vals is None else torch.cat((vals,cur_vals),0)
		# compute mean val
		cur_psnr_tensor = vals.mean()
		cur_psnr = cur_psnr_tensor.item()
		# compare current score
		if self.best_psnr is None or (cur_psnr > self.best_psnr and not torch.isinf(cur_psnr_tensor)):
			# update best
			self.best_psnr = cur_psnr
			# save mode
			self.save(best=True)
		# print score
		print('\n### Current PSNR: {:.4f} (best: {:.4f}) ###\n'.format(cur_psnr,self.best_psnr))
		# set back gen in train mode
		self.netG.train()
		# return score
		return cur_psnr


	def regen(self):
		# load
		self.load(best=True)
		# set net in evaluation mode
		self.netG.eval()
		# move net in device
		self.netG.to(self.device)
		# create dataloader
		data_loader = data.DataLoader(
				get_dataset(self.params['dataset'],phase='regen'),
				batch_size = self.params['dataset']['params']['regen']['batch_size'],
				num_workers = self.params['dataset']['params']['regen']['num_workers'],
				shuffle = False,
		)
		# regenerate
		for inp, fn in tqdm(data_loader):
			# move to device
			inp = inp.to(self.device)
			# regenerate image
			with torch.no_grad():
				regen = self.netG(inp).cpu()
			# save images
			for i in range(len(fn)):
				cur_fn = os.path.join(self.out_dir,fn[i])
				transforms.ToPILImage()(torch.clamp(regen[i],0,1)).save(cur_fn)


def update_dict(d, u):
	for k, v in u.items():
		if isinstance(v, collections.Mapping):
			d[k] = update_dict(d.get(k, {}), v)
		else:
			d[k] = v
	return d


if __name__ == '__main__':
	# parse args
	parser = argparse.ArgumentParser(description='Paired Training.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# json
	parser.add_argument("-json", "--json", help="JSON containing parameters", nargs='+', required=True)
	parser.add_argument("-r", "--regen",  help="Regen images", action='store_true')
	# parse arguments
	args = parser.parse_args()
	# join all dicts
	params = {}
	for i in range(len(args.json)):
		# load current json
		with open(args.json[i]) as f: cur_params = json.load(f)
		# merge with other json files
		if i==0:
			params = cur_params.copy()
		else:
			params = update_dict(params, cur_params)
	# define expname as json file name
	exp_name = '_'.join([os.path.splitext(os.path.basename(curj))[0] for curj in args.json])
	# set exp name in params
	params['exp_name'] = exp_name
	# create trainer
	trainer = Trainer(params)
	# train or regen
	if args.regen:
		trainer.regen()
	else:
		trainer.train()
