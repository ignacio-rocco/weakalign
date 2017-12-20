from __future__ import print_function, division
import argparse
import os
from os.path import exists, join, basename
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
#from torch.utils.data import DataLoader
from util.dataloader import DataLoader # modified dataloader
from model.cnn_geometric_model import CNNGeometric, TwoStageCNNGeometric, FeatureCorrelation, featureL2Norm
from model.loss import TransformedGridLoss, WeakInlierCount, TwoStageWeakInlierCount
from data.synth_dataset import SynthDataset
from data.weak_dataset import ImagePairDataset
from data.pf_dataset import PFDataset, PFPascalDataset
from data.download_datasets import download_PF_pascal
from geotnf.transformation import SynthPairTnf,SynthTwoPairTnf,SynthTwoStageTwoPairTnf
from image.normalization import NormalizeImageDict
from util.torch_util import save_checkpoint, str_to_bool
from util.torch_util import BatchTensorToVars
from geotnf.transformation import GeometricTnf
from collections import OrderedDict
import numpy as np
import numpy.random
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import torch.nn.functional as F
from model.cnn_geometric_model import featureL2Norm
from util.dataloader import default_collate
from util.eval_util import pck_metric, area_metrics, flow_metrics, compute_metric
from options.options import ArgumentParser


"""

Script to train the model using weak supervision

"""

print('WeakAlign training script using weak supervision')

# Argument parsing
args,arg_groups = ArgumentParser(mode='train_weak').parse()
print(args)

use_cuda = torch.cuda.is_available()

# Seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# CNN model and loss
print('Creating CNN model...')

model = TwoStageCNNGeometric(use_cuda=use_cuda,
                             return_correlation=True,
                             **arg_groups['model'])

# Download validation dataset if needed
if args.eval_dataset_path=='' and args.eval_dataset=='pf-pascal':
    args.eval_dataset_path='datasets/proposal-flow-pascal/'
if args.eval_dataset=='pf-pascal' and not exists(args.eval_dataset_path):
    download_PF_pascal(args.eval_dataset_path)

# load pre-trained model
if args.model!='':
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
        
    for name, param in model.FeatureExtraction.state_dict().items():
        model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    
    for name, param in model.FeatureRegression.state_dict().items():
        model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])
    for name, param in model.FeatureRegression2.state_dict().items():
        model.FeatureRegression2.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression2.' + name])
        
if args.model_aff!='':
    checkpoint_aff = torch.load(args.model_aff, map_location=lambda storage, loc: storage)
    checkpoint_aff['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint_aff['state_dict'].items()])
    for name, param in model.FeatureExtraction.state_dict().items():
        model.FeatureExtraction.state_dict()[name].copy_(checkpoint_aff['state_dict']['FeatureExtraction.' + name])    
    for name, param in model.FeatureRegression.state_dict().items():
        model.FeatureRegression.state_dict()[name].copy_(checkpoint_aff['state_dict']['FeatureRegression.' + name])

if args.model_tps!='':
    checkpoint_tps = torch.load(args.model_tps, map_location=lambda storage, loc: storage)
    checkpoint_tps['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint_tps['state_dict'].items()])
    for name, param in model.FeatureRegression2.state_dict().items():
        model.FeatureRegression2.state_dict()[name].copy_(checkpoint_tps['state_dict']['FeatureRegression.' + name])
        
# set which parts of model to train      
for name,param in model.FeatureExtraction.named_parameters():
    param.requires_grad = False
    if args.train_fe and np.sum([name.find(x)!=-1 for x in args.fe_finetune_params]):
        param.requires_grad = True        
    if args.train_fe and name.find('bn')!=-1 and np.sum([name.find(x)!=-1 for x in args.fe_finetune_params]):
         param.requires_grad = args.train_bn 
            
for name,param in model.FeatureExtraction.named_parameters():
    print(name.ljust(30),param.requires_grad)
        
for name,param in model.FeatureRegression.named_parameters():    
    param.requires_grad = args.train_fr 
    if args.train_fr and name.find('bn')!=-1:
        param.requires_grad = args.train_bn            

for name,param in model.FeatureRegression2.named_parameters():    
    param.requires_grad = args.train_fr 
    if args.train_fr and name.find('bn')!=-1:
        param.requires_grad = args.train_bn

# define loss
print('Using weak loss...')
if args.dilation_filter==0:
    dilation_filter = 0
else:
    dilation_filter = generate_binary_structure(2, args.dilation_filter)
        
inliersAffine = WeakInlierCount(geometric_model='affine',**arg_groups['weak_loss'])
inliersTps = WeakInlierCount(geometric_model='tps',**arg_groups['weak_loss'])
inliersComposed = TwoStageWeakInlierCount(use_cuda=use_cuda,**arg_groups['weak_loss'])


def inlier_score_function(theta_aff,theta_aff_tps,corr_aff,corr_aff_tps,minimize_outliers=False):
    inliers_comp = inliersComposed(matches=corr_aff,
                                                 theta_aff=theta_aff,
                                                 theta_aff_tps=theta_aff_tps)
    
    inliers_aff = inliersAffine(matches=corr_aff,
                                theta=theta_aff)
    
    inlier_score=inliers_aff+inliers_comp
    
    return inlier_score

def loss_fun(batch):
    
    theta_aff,theta_aff_tps,corr_aff,corr_aff_tps=model(batch)
    
    inlier_score_pos = inlier_score_function(theta_aff,
                                             theta_aff_tps,
                                             corr_aff,
                                             corr_aff_tps)
    loss = torch.mean(-inlier_score_pos)

    return loss

# dataset 
train_dataset_size = args.train_dataset_size if args.train_dataset_size!=0 else None

dataset = ImagePairDataset(csv_file=os.path.join(args.dataset_csv_path,'train_pairs.csv'),
                       training_image_path=args.dataset_image_path,
                       transform=NormalizeImageDict(['source_image','target_image']),
                       dataset_size = train_dataset_size,
                       random_crop=args.random_crop)

dataset_eval = PFPascalDataset(csv_file=os.path.join(args.eval_dataset_path, 'val_pairs_pf_pascal.csv'),
                      dataset_path=args.eval_dataset_path,
                      transform=NormalizeImageDict(['source_image','target_image']))

# filter training categories
if args.categories!=0:
    keep = np.zeros((len(dataset.set),1))
    for i in range(len(dataset.set)):
        keep[i]=np.sum(dataset.set[i]==args.categories)
    keep_idx = np.nonzero(keep)[0]
    dataset.set = dataset.set[keep_idx]
    dataset.img_A_names = dataset.img_A_names[keep_idx]
    dataset.img_B_names = dataset.img_B_names[keep_idx]

batch_tnf = BatchTensorToVars(use_cuda=use_cuda)

# dataloader
dataloader = DataLoader(dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=4)

dataloader_eval = DataLoader(dataset_eval, batch_size=8,
                        shuffle=False, num_workers=4)

# define checkpoint name
checkpoint_suffix = '_' + args.feature_extraction_cnn
if args.tps_reg_factor != 0:
    checkpoint_suffix += '_regfact' + str(args.tps_reg_factor)

checkpoint_name = os.path.join(args.result_model_dir,
                               args.result_model_fn + checkpoint_suffix + '.pth.tar')
print(checkpoint_name)

# define epoch function
def process_epoch(mode,epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,use_cuda=True,log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if mode=='train':
            optimizer.zero_grad()
        tnf_batch = batch_preprocessing_fn(batch)
        loss = loss_fn(tnf_batch)
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

# compute initial value of evaluation metric used for early stopping
if args.eval_metric=='dist':
    metric = 'dist'
if args.eval_metric=='pck':
    metric = 'pck'
do_aff = args.model_aff!=""
do_tps = args.model_tps!=""
two_stage = args.model!='' or (do_aff and do_tps)


if args.categories==0: 
    eval_categories = np.array(range(20))+1
else:
    eval_categories = np.array(args.categories)
    
eval_flag = np.zeros(len(dataset_eval))
for i in range(len(dataset_eval)):
    eval_flag[i]=sum(eval_categories==dataset_eval.category[i])
eval_idx = np.flatnonzero(eval_flag)

model.eval()

stats=compute_metric(metric,model,dataset_eval,dataloader_eval,batch_tnf,8,two_stage,do_aff,do_tps,args)
eval_value=np.mean(stats['aff_tps'][metric][eval_idx])

print(eval_value)

# train
best_test_loss = float("inf")

train_loss = np.zeros(args.num_epochs)
test_loss = np.zeros(args.num_epochs)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

print('Starting training...')

for epoch in range(1, args.num_epochs+1):
    if args.update_bn_buffers==False:
        model.eval()
    else:
        model.train()
    train_loss[epoch-1] = process_epoch('train',epoch,model,loss_fun,optimizer,dataloader,batch_tnf,log_interval=1)
    model.eval()
    stats=compute_metric(metric,model,dataset_eval,dataloader_eval,batch_tnf,8,two_stage,do_aff,do_tps,args)
    eval_value=np.mean(stats['aff_tps'][metric][eval_idx])
    print(eval_value)
    
    if args.eval_metric=='pck':
        test_loss[epoch-1] = -eval_value
    else:
        test_loss[epoch-1] = eval_value
        
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