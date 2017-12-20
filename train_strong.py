from __future__ import print_function, division
import os
from os.path import exists, join, basename
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
from util.dataloader import DataLoader # modified dataloader
from torch.utils.data import Dataset
from model.cnn_geometric_model import CNNGeometric
from model.loss import TransformedGridLoss
from data.synth_dataset import SynthDataset
from data.download_datasets import download_pascal
from geotnf.transformation import SynthPairTnf
from image.normalization import NormalizeImageDict
#from util.train_test_fn import train_fun_strong, test_fun_strong
from util.torch_util import save_checkpoint, str_to_bool
import numpy as np
import numpy.random
from collections import OrderedDict
from options.options import ArgumentParser

"""

Script to train the model as presented in the CNNGeometric CVPR'17 paper
using synthetically warped image pairs and strong supervision

"""

print('CNNGeometric training script using strong supervision')

# Argument parsing
args,arg_groups = ArgumentParser(mode='train_strong').parse()
print(args)

# Seed and CUDA
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# Download dataset if needed and set paths
if args.training_dataset == 'pascal':
    if args.dataset_image_path == '':
        if not exists('datasets/pascal-voc11/TrainVal'):
            download_pascal('datasets/pascal-voc11/')
        args.dataset_image_path = 'datasets/pascal-voc11/'        
    if args.dataset_csv_path == '' and args.geometric_model=='affine':
        args.dataset_csv_path = 'training_data/pascal-synth-aff'
    elif args.dataset_csv_path == '' and args.geometric_model=='tps':
        args.dataset_csv_path = 'training_data/pascal-synth-tps'

arg_groups['dataset']['dataset_image_path']=args.dataset_image_path

# CNN model and loss
print('Creating CNN model...')
if args.geometric_model=='affine':
    cnn_output_dim = 6
elif args.geometric_model=='tps':
    cnn_output_dim = 18

model = CNNGeometric(use_cuda=use_cuda,
                     output_dim=cnn_output_dim,
                     **arg_groups['model'])

# Load pretrained model
if args.model!='':
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        
    for name, param in model.FeatureExtraction.state_dict().items():
        model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    
    for name, param in model.FeatureRegression.state_dict().items():
        model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])
        

if args.use_mse_loss:
    print('Using MSE loss...')
    loss = nn.MSELoss()
else:
    print('Using grid loss...')
    loss = TransformedGridLoss(use_cuda=use_cuda,geometric_model=args.geometric_model)

# Dataset and dataloader
dataset = SynthDataset(geometric_model=args.geometric_model,
                       transform=NormalizeImageDict(['image']),
                       dataset_csv_file = 'train.csv',
                       **arg_groups['dataset'])

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, 
                        num_workers=4) # don't change num_workers, as they copy the rnd seed

dataset_test = SynthDataset(geometric_model=args.geometric_model,
                            transform=NormalizeImageDict(['image']),
                            dataset_csv_file='test.csv',
                            **arg_groups['dataset'])

dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

cnn_image_size=(args.image_size,args.image_size)

pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model,
                                   output_size=cnn_image_size,
                                   use_cuda=use_cuda)

# Optimizer
optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=args.lr)
    
# Define checkpoint name
checkpoint_suffix = '_strong'
checkpoint_suffix += '_' + str(args.num_epochs)
checkpoint_suffix += '_' + args.training_dataset
checkpoint_suffix += '_' + args.geometric_model
checkpoint_suffix += '_' + args.feature_extraction_cnn
if args.use_mse_loss:
    checkpoint_suffix += '_mse_loss'
else:
    checkpoint_suffix += '_grid_loss'

checkpoint_name = os.path.join(args.result_model_dir,
                               args.result_model_fn + checkpoint_suffix + '.pth.tar')

print(checkpoint_name)    
    
# Train
best_test_loss = float("inf")

# define epoch function
def process_epoch(mode,epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,use_cuda=True,log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if mode=='train':
            optimizer.zero_grad()
        tnf_batch = batch_preprocessing_fn(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta,tnf_batch['theta_GT'])
        loss_np = loss.data.cpu().numpy()[0]
        epoch_loss += loss_np
        if mode=='train':
            loss.backward()
            optimizer.step()
        else:
            loss=None
        if batch_idx % log_interval == 0:
            print(mode.capitalize()+' Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx , len(dataloader),
                100. * batch_idx / len(dataloader), loss_np))
    epoch_loss /= len(dataloader)
    print(mode.capitalize()+' set: Average loss: {:.4f}'.format(epoch_loss))
    return epoch_loss


train_loss = np.zeros(args.num_epochs)
test_loss = np.zeros(args.num_epochs)

print('Starting training...')

model.FeatureExtraction.eval()

for epoch in range(1, args.num_epochs+1):
    model.FeatureRegression.train()
    train_loss[epoch-1] = process_epoch('train',epoch,model,loss,optimizer,dataloader,pair_generation_tnf,log_interval=100)
    model.FeatureRegression.eval()
    test_loss[epoch-1] = process_epoch('test',epoch,model,loss,optimizer,dataloader_test,pair_generation_tnf,log_interval=100)
    
    # remember best loss
    is_best = test_loss[epoch-1] < best_test_loss
    best_test_loss = min(test_loss[epoch-1], best_test_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'args': args,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer' : optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, is_best,checkpoint_name)

print('Done!')