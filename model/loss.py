from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from util.torch_util import expand_dim
from geotnf.transformation import GeometricTnf,ComposedGeometricTnf
import torch.nn.functional as F
import scipy.signal

class TransformedGridLoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True, grid_size=20):
        super(TransformedGridLoss, self).__init__()
        self.geometric_model = geometric_model
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(axis_coords,axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P),requires_grad=False)
        self.pointTnf = PointTnf(use_cuda=use_cuda)
        if use_cuda:
            self.P = self.P.cuda();

    def forward(self, theta, theta_GT):
        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size,2,self.N)
        # compute transformed grid points using estimated and GT tnfs
        if self.geometric_model=='affine':
            P_prime = self.pointTnf.affPointTnf(theta,P)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT,P)
        elif self.geometric_model=='tps':
            P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3),P)
            P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT,P)
        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_GT,2),1)
        loss = torch.mean(loss)
        return loss

class WeakInlierCount(nn.Module):
    def __init__(self, geometric_model='affine', tps_grid_size=3, tps_reg_factor=0, h_matches=15, w_matches=15, use_conv_filter=False, dilation_filter=None, use_cuda=True, normalize_inlier_count=False, offset_factor=227/210):
        super(WeakInlierCount, self).__init__()
        self.normalize=normalize_inlier_count
        self.geometric_model = geometric_model
        self.geometricTnf = GeometricTnf(geometric_model=geometric_model,
                                         tps_grid_size=tps_grid_size,
                                         tps_reg_factor=tps_reg_factor,
                                         out_h=h_matches, out_w=w_matches,
                                         offset_factor = offset_factor,
                                         use_cuda=use_cuda)
        # define dilation filter
        if dilation_filter is None:
            dilation_filter = generate_binary_structure(2, 2)
        # define identity mask tensor (w,h are switched and will be permuted back later)
        mask_id = np.zeros((w_matches,h_matches,w_matches*h_matches))
        idx_list = list(range(0, mask_id.size, mask_id.shape[2]+1))
        mask_id.reshape((-1))[idx_list]=1
        mask_id = mask_id.swapaxes(0,1)
        # perform 2D dilation to each channel 
        if not use_conv_filter:
            if not (isinstance(dilation_filter,int) and dilation_filter==0):
                for i in range(mask_id.shape[2]):
                    mask_id[:,:,i] = binary_dilation(mask_id[:,:,i],structure=dilation_filter).astype(mask_id.dtype)
        else:
            for i in range(mask_id.shape[2]):
                flt=np.array([[1/16,1/8,1/16],
                                 [1/8, 1/4, 1/8],
                                 [1/16,1/8,1/16]])
                mask_id[:,:,i] = scipy.signal.convolve2d(mask_id[:,:,i], flt, mode='same', boundary='fill', fillvalue=0)
            
        # convert to PyTorch variable
        mask_id = Variable(torch.FloatTensor(mask_id).transpose(1,2).transpose(0,1).unsqueeze(0),requires_grad=False)
        self.mask_id = mask_id
        if use_cuda:
            self.mask_id = self.mask_id.cuda();

    def forward(self, theta, matches, return_outliers=False):
        if isinstance(theta,Variable): # handle normal batch transformations
            batch_size=theta.size()[0]
            theta=theta.clone()
            mask = self.geometricTnf(expand_dim(self.mask_id,0,batch_size),theta)
            if return_outliers:
                mask_outliers = self.geometricTnf(expand_dim(1.0-self.mask_id,0,batch_size),theta)
            if self.normalize:
                epsilon=1e-5
                mask = torch.div(mask,
                                 torch.sum(torch.sum(torch.sum(mask+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask))
                if return_outliers:
                    mask_outliers = torch.div(mask_outliers,
                                              torch.sum(torch.sum(torch.sum(mask_outliers+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask_outliers))
            score = torch.sum(torch.sum(torch.sum(torch.mul(mask,matches),3),2),1)
            if return_outliers:
                score_outliers = torch.sum(torch.sum(torch.sum(torch.mul(mask_outliers,matches),3),2),1)
                return (score,score_outliers)
        elif isinstance(theta,list): # handle multiple transformations per batch item, batch is in list format (used for RANSAC)
            batch_size = len(theta)
            score = []
            for b in range(batch_size):
                sample_size=theta[b].size(0)
                s=self.forward(theta[b],expand_dim(matches[b,:,:,:].unsqueeze(0),0,sample_size))
                score.append(s)
        return score
    
class TwoStageWeakInlierCount(WeakInlierCount):
    def __init__(self, 
                 tps_grid_size=3,
                 tps_reg_factor=0,
                 h_matches=15,
                 w_matches=15,
                 use_conv_filter=False,
                 dilation_filter=None,
                 use_cuda=True,
                 normalize_inlier_count=False,
                 offset_factor=227/210):
        
        super(TwoStageWeakInlierCount, self).__init__(h_matches=h_matches,
                                                      w_matches=w_matches,
                                                      use_conv_filter=use_conv_filter,
                                                      dilation_filter=dilation_filter,
                                                      use_cuda=use_cuda,
                                                      normalize_inlier_count=normalize_inlier_count,
                                                      offset_factor=offset_factor)
        
        self.compGeometricTnf = ComposedGeometricTnf(tps_grid_size=tps_grid_size,
                                                     tps_reg_factor=tps_reg_factor,
                                                     out_h=h_matches,
                                                     out_w=w_matches,
                                                     offset_factor=offset_factor,
                                                     use_cuda=use_cuda)
        
    def forward(self, theta_aff, theta_aff_tps, matches,return_outliers=False):
        batch_size=theta_aff.size()[0]
        mask = self.compGeometricTnf(image_batch=expand_dim(self.mask_id,0,batch_size),
                                     theta_aff=theta_aff,
                                     theta_aff_tps=theta_aff_tps)
        if return_outliers:
             mask_outliers = self.compGeometricTnf(image_batch=expand_dim(1.0-self.mask_id,0,batch_size),
                                                   theta_aff=theta_aff,
                                                   theta_aff_tps=theta_aff_tps)
        if self.normalize:
            epsilon=1e-5
            mask = torch.div(mask,
                             torch.sum(torch.sum(torch.sum(mask+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask))
            if return_outliers:
                mask_outliers = torch.div(mask,
                             torch.sum(torch.sum(torch.sum(mask_outliers+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask_outliers)) 
        score = torch.sum(torch.sum(torch.sum(torch.mul(mask,matches),3),2),1)

        if return_outliers:
            score_outliers = torch.sum(torch.sum(torch.sum(torch.mul(mask_outliers,matches),3),2),1)
            return (score,score_outliers)
        return score