import os
import math
import torch
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.cm
import torchvision.utils as vutils
from models.loss import Sobel
import matplotlib.pyplot as plt
cmap = plt.cm.viridis


def draw_losses(logger, loss, global_step):
	name = "train_loss"
	logger.add_scalar(name, loss, global_step)

def draw_images(logger, all_draw_image, global_step):
	for image_name, images in all_draw_image.items():
		if images.shape[1] == 1:
			images = colormap(images)
		elif images.shape[1] == 3:
			__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
						'std': [0.229, 0.224, 0.225]}
			for channel in np.arange(images.shape[1]):
				images[:, channel, :, :]  = images[:, channel, :, :] * __imagenet_stats["std"][channel] + __imagenet_stats["mean"][channel] 

		if len(images.shape) == 3:
			images = images[np.newaxis, :, :, :] 
		if images.shape[0]>4:
			images = images[:4, :, :, :]
        
		logger.add_image(image_name, images, global_step)

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb              = np.transpose(input.cpu().numpy(), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu   = np.squeeze(depth_pred.data.cpu().numpy()) 

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    # img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    # return img_merge
    return rgb, depth_target_col, depth_pred_col


def makedir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def adjust_learning_rate(optimizer, epoch, init_lr):

	lr = init_lr * (0.1 ** (epoch // 5))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def save_checkpoint(state, filename):
	torch.save(state, filename)


def edge_detection(depth):
	get_edge = Sobel().cuda()

	edge_xy = get_edge(depth)
	edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
		torch.pow(edge_xy[:, 1, :, :], 2)
	edge_sobel = torch.sqrt(edge_sobel)

	return edge_sobel

def build_optimizer(model,
					learning_rate, 
					optimizer_name='rmsprop',
					weight_decay=1e-5,
					epsilon=0.001,
					momentum=0.9):
	"""Build optimizer"""
	if optimizer_name == "sgd":
		print("Using SGD optimizer.")
		optimizer = torch.optim.SGD(model.parameters(), 
									lr = learning_rate,
									momentum=momentum,
									weight_decay=weight_decay)

	elif optimizer_name	== 'rmsprop':
		print("Using RMSProp optimizer.")
		optimizer = torch.optim.RMSprop(model.parameters(),
										lr = learning_rate,
										eps = epsilon,
										weight_decay = weight_decay,
										momentum = momentum
										)
	elif optimizer_name == 'adam':
		print("Using Adam optimizer.")
		optimizer = torch.optim.Adam(model.parameters(), 
									 lr = learning_rate, weight_decay=weight_decay)
	return optimizer




#original script: https://github.com/fangchangma/sparse-to-dense/blob/master/utils.lua
	

def lg10(x):
	return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
	z = x.clone()
	maskYLarger = torch.lt(x, y)
	z[maskYLarger.detach()] = y[maskYLarger.detach()]
	return z

def nValid(x):
	return torch.sum(torch.eq(x, x).float())

def nValid_alt(x):
	return torch.sum(torch.gt(x, 0.).float())

def nNanElement(x):
	return torch.sum(torch.ne(x, x).float())

def getNanMask(x):
	return torch.ne(x, x)

def getNanMask_alt(x):
	return torch.eq(x, 0.)

def setNanToZero(input, target):
	nanMask = getNanMask(target)
	# nValidElement = nValid(target)

	_input = input.clone()
	_target = target.clone()

	_input[nanMask] = 0
	_target[nanMask] = 0
	_input[torch.eq(_target, 0)] = 0
	_input[getNanMask(input)] = 0


	nValidElement = nValid_alt(_target)
	nanMask = getNanMask_alt(_target)

	return _input, _target, nanMask, nValidElement


def evaluateError(output, target):
	errors = {'SQ_REL': 0, 'RMSE': 0, 'ABS_REL': 0, 'RMSEl': 0,
			  'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

	_output, _target, nanMask, nValidElement = setNanToZero(output, target)

	if (nValidElement.data.cpu().numpy() > 0):
		# perform median scaling
		b = _target.size(dim=0)
		ratio =  _target[torch.logical_not(nanMask)].reshape([-1]).median()/_output[torch.logical_not(nanMask)].reshape([-1]).median()
		_output *= ratio

		_output = torch.clamp(_output, 0.001, 80.0)
		_target = torch.clamp(_target, 0.001, 80.0)

		diffMatrix = torch.abs(_output - _target)

		errors['RMSE'] = torch.sqrt(torch.sum(torch.pow(diffMatrix, 2)) / nValidElement)
		errors['SQ_REL'] = torch.sum(torch.pow(diffMatrix, 2)/(_output+0.0000001)) / nValidElement

		errors['MAE'] = torch.sum(diffMatrix) / nValidElement

		realMatrix = torch.div(diffMatrix, _target)
		realMatrix[nanMask] = 0
		errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement

		RMSElMatrix = torch.abs(torch.log(_output) - torch.log(_target))

		RMSElMatrix[nanMask] = 0
		errors['RMSEl'] = torch.sqrt(torch.sum(RMSElMatrix) / nValidElement)

		yOverZ = torch.div(_output, _target)
		zOverY = torch.div(_target, _output)

		maxRatio = maxOfTwo(yOverZ, zOverY)

		errors['DELTA1'] = torch.sum(
			torch.le(maxRatio, 1.25).float()) / nValidElement
		errors['DELTA2'] = torch.sum(
			torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
		errors['DELTA3'] = torch.sum(
			torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement

		errors['RMSE'] = float(errors['RMSE'].data.cpu().numpy())
		errors['SQ_REL'] = float(errors['SQ_REL'].data.cpu().numpy())
		errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
		errors['RMSEl'] = float(errors['RMSEl'].data.cpu().numpy())
		errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
		errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
		errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
		errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())

	return errors


def addErrors(errorSum, errors, batchSize):
	errorSum['RMSE']=errorSum['RMSE'] + errors['RMSE'] * batchSize
	errorSum['SQ_REL']=errorSum['SQ_REL'] + errors['SQ_REL'] * batchSize
	errorSum['ABS_REL']=errorSum['ABS_REL'] + errors['ABS_REL'] * batchSize
	errorSum['RMSEl']=errorSum['RMSEl'] + errors['RMSEl'] * batchSize
	errorSum['MAE']=errorSum['MAE'] + errors['MAE'] * batchSize

	errorSum['DELTA1']=errorSum['DELTA1'] + errors['DELTA1'] * batchSize
	errorSum['DELTA2']=errorSum['DELTA2'] + errors['DELTA2'] * batchSize
	errorSum['DELTA3']=errorSum['DELTA3'] + errors['DELTA3'] * batchSize

	return errorSum


def averageErrors(errorSum, N):
	averageError={'SQ_REL': 0, 'RMSE': 0, 'ABS_REL': 0, 'RMSEl': 0,
					'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

	averageError['RMSE'] = errorSum['RMSE'] / N
	averageError['SQ_REL'] = errorSum['SQ_REL'] / N
	averageError['ABS_REL'] = errorSum['ABS_REL'] / N
	averageError['RMSEl'] = errorSum['RMSEl'] / N
	averageError['MAE'] = errorSum['MAE'] / N

	averageError['DELTA1'] = errorSum['DELTA1'] / N
	averageError['DELTA2'] = errorSum['DELTA2'] / N
	averageError['DELTA3'] = errorSum['DELTA3'] / N

	return averageError


def colormap(image, cmap="jet"):
	image_min = torch.min(image)
	image_max = torch.max(image)
	image = (image - image_min) / (image_max - image_min)
	image = torch.squeeze(image)

	# quantize 
	indices = torch.round(image * 255).long()
	# gather
	cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')

	colors = cm(np.arange(256))[:, :3]
	colors = torch.cuda.FloatTensor(colors)
	color_map = colors[indices].transpose(2, 3).transpose(1, 2)

	return color_map







	
