from __future__ import print_function, division
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.cnn_geometric_model import CNNGeometric,TwoStageCNNGeometric
from data.pf_dataset import PFDataset, PFPascalDataset
from data.pascal_parts_dataset import PascalPartsDataset
from data.caltech_dataset import CaltechDataset
from data.tss_dataset import TSSDataset
from data.download_datasets import *
from image.normalization import NormalizeImageDict
from util.torch_util import BatchTensorToVars, str_to_bool
from geotnf.point_tnf import *
from geotnf.transformation import GeometricTnf
from os.path import exists
from util.eval_util import pck_metric, area_metrics, flow_metrics, compute_metric
from collections import OrderedDict
from options.options import ArgumentParser
from util.dataloader import default_collate
from util.torch_util import collate_custom

"""

Script to evaluate a trained model as presented in the CNNGeometric CVPR'17 paper
on the ProposalFlow/Caltech-101 dataset

"""

print('WeakAlign evaluation script')

# Argument parsing
args,arg_groups = ArgumentParser(mode='eval').parse()
print(args)

# check provided models and deduce if single/two-stage model should be used
do_aff = args.model_aff!=""
do_tps = args.model_tps!=""
two_stage = args.model!='' or (do_aff and do_tps)

if args.eval_dataset_path=='' and args.eval_dataset=='pf':
    args.eval_dataset_path='datasets/proposal-flow-willow/'

if args.eval_dataset_path=='' and args.eval_dataset=='pf-pascal':
    args.eval_dataset_path='datasets/proposal-flow-pascal/'

if args.eval_dataset_path=='' and args.eval_dataset=='caltech':
    args.eval_dataset_path='datasets/caltech-101/'
    
if args.eval_dataset_path=='' and args.eval_dataset=='tss':
    args.eval_dataset_path='datasets/tss/'

if args.eval_dataset_path=='' and args.eval_dataset=='pascal-parts':
    args.eval_dataset_path='datasets/pascal-parts/'

use_cuda = torch.cuda.is_available()

# Download dataset if needed
if args.eval_dataset=='pf' and not exists(args.eval_dataset_path):
    download_PF_willow(args.eval_dataset_path)
elif args.eval_dataset=='pf-pascal' and not exists(args.eval_dataset_path):
    download_PF_pascal(args.eval_dataset_path)
elif args.eval_dataset=='caltech' and not exists(args.eval_dataset_path):
    download_caltech(args.eval_dataset_path)
elif args.eval_dataset=='tss' and not exists(args.eval_dataset_path):
    download_TSS(args.eval_dataset_path)
elif args.eval_dataset=='pascal-parts' and not exists(args.eval_dataset_path):
    download_pascal_parts(args.eval_dataset_path)

# Create model
print('Creating CNN model...')
# check type of given model and create model
if two_stage:
    model = TwoStageCNNGeometric(use_cuda=use_cuda,
                                 **arg_groups['model'])
if not two_stage and do_aff:
    model = CNNGeometric(use_cuda=use_cuda,
                         output_dim=6,
                         **arg_groups['model'])
    
if not two_stage and do_tps:
    model_tps = CNNGeometric(use_cuda=use_cuda,
                             output_dim=18,
                             **arg_groups['model'])

# load pretrained weights
if two_stage and args.model!='':
        checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
        checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])

        for name, param in model.FeatureExtraction.state_dict().items():
            model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    
        for name, param in model.FeatureRegression.state_dict().items():
            model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])
        for name, param in model.FeatureRegression2.state_dict().items():
            model.FeatureRegression2.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression2.' + name])
            
if two_stage and args.model=='':
    checkpoint_aff = torch.load(args.model_aff, map_location=lambda storage, loc: storage)
    checkpoint_aff['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint_aff['state_dict'].items()])

    checkpoint_tps = torch.load(args.model_tps, map_location=lambda storage, loc: storage)
    checkpoint_tps['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint_tps['state_dict'].items()])

    for name, param in model.FeatureRegression.state_dict().items():
        model.FeatureRegression.state_dict()[name].copy_(checkpoint_aff['state_dict']['FeatureRegression.' + name])
    for name, param in model.FeatureRegression2.state_dict().items():
        model.FeatureRegression2.state_dict()[name].copy_(checkpoint_tps['state_dict']['FeatureRegression.' + name])
        
if not two_stage:
    model_fn = args.model_aff if do_aff else args.model_tps
    checkpoint = torch.load(model_fn, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
    for name, param in model.FeatureExtraction.state_dict().items():
        model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])    
    for name, param in model.FeatureRegression.state_dict().items():
        model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])

# Dataset and dataloader
if args.eval_dataset=='pf':  
    Dataset = PFDataset
    collate_fn = default_collate
    csv_file = 'test_pairs_pf.csv'
if args.eval_dataset=='pf-pascal':  
    Dataset = PFPascalDataset
    collate_fn = default_collate
    csv_file = 'test_pairs_pf_pascal.csv'    
elif args.eval_dataset=='caltech':
    Dataset = CaltechDataset
    collate_fn = default_collate
    csv_file = 'test_pairs_caltech_with_category.csv'
elif args.eval_dataset=='tss':
    Dataset = TSSDataset
    collate_fn = default_collate
    csv_file = 'test_pairs_tss.csv'
elif args.eval_dataset=='pascal-parts':
    Dataset = PascalPartsDataset
    collate_fn = collate_custom
    csv_file = 'test_pairs_pascal_parts.csv'
    
cnn_image_size=(args.image_size,args.image_size)

dataset = Dataset(csv_file=os.path.join(args.eval_dataset_path, csv_file),
                  dataset_path=args.eval_dataset_path,
                  transform=NormalizeImageDict(['source_image','target_image']),
                  output_size=cnn_image_size)

if use_cuda:
    batch_size=8
else:
    batch_size=1

dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4,
                        collate_fn=collate_fn)

batch_tnf = BatchTensorToVars(use_cuda=use_cuda)


if args.eval_dataset=='pf' or args.eval_dataset=='pf-pascal':  
    metric = 'pck'
elif args.eval_dataset=='caltech':
    metric = 'area'
elif args.eval_dataset=='pascal-parts':
    metric = 'pascal_parts'
elif args.eval_dataset=='tss':
    metric = 'flow'
    
model.eval()
    
stats=compute_metric(metric,model,dataset,dataloader,batch_tnf,batch_size,two_stage,do_aff,do_tps,args)

