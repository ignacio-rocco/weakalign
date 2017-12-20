from __future__ import print_function, division
import torch
import os
from os.path import exists, join, basename
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable


class SynthDataset(Dataset):
    """
    
    Synthetically transformed pairs dataset for training with strong supervision
    
    Args:
            csv_file (string): Path to the csv file with image names and transformations.
            training_image_path (string): Directory with all the images.
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)
            
    Returns:
            Dict: {'image': full dataset image, 'theta': desired transformation}
            
    """

    def __init__(self, dataset_csv_path, dataset_csv_file, dataset_image_path, output_size=(480,640), geometric_model='affine', dataset_size=0, transform=None, random_sample=False, random_t=0.5, random_s=0.5, random_alpha=1/6, random_t_tps=0.4):
        self.out_h, self.out_w = output_size
        # read csv file
        self.train_data = pd.read_csv(os.path.join(dataset_csv_path,dataset_csv_file))
        self.random_sample = random_sample
        self.random_t = random_t
        self.random_t_tps = random_t_tps
        self.random_alpha = random_alpha
        self.random_s = random_s
        if dataset_size!=0:
            dataset_size = min((dataset_size,len(self.train_data)))
            self.train_data = self.train_data.iloc[0:dataset_size,:]
        self.img_names = self.train_data.iloc[:,0]
        if self.random_sample==False:
            self.theta_array = self.train_data.iloc[:, 1:].as_matrix().astype('float')
        # copy arguments
        self.dataset_image_path = dataset_image_path
        self.transform = transform
        self.geometric_model = geometric_model
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # read image
        img_name = os.path.join(self.dataset_image_path, self.img_names[idx])
        image = io.imread(img_name)
        
        # read theta
        if self.random_sample==False:
            theta = self.theta_array[idx, :]

            if self.geometric_model=='affine':
                # reshape theta to 2x3 matrix [A|t] where 
                # first row corresponds to X and second to Y
    #            theta = theta[[0,1,4,2,3,5]].reshape(2,3)
                theta = theta[[3,2,5,1,0,4]] #.reshape(2,3)
            if self.geometric_model=='tps':
                theta = np.expand_dims(np.expand_dims(theta,1),2)
            if self.geometric_model=='afftps':
                theta[[0,1,2,3,4,5]] = theta[[3,2,5,1,0,4]]
        else:
            if self.geometric_model=='affine' or self.geometric_model=='afftps':
                alpha = (np.random.rand(1)-0.5)*2*np.pi*self.random_alpha
                theta_aff = np.random.rand(6)
                theta_aff[[2,5]]=(theta_aff[[2,5]]-0.5)*2*self.random_t
                theta_aff[0]=(1+(theta_aff[0]-0.5)*2*self.random_s)*np.cos(alpha)
                theta_aff[1]=(1+(theta_aff[1]-0.5)*2*self.random_s)*(-np.sin(alpha))
                theta_aff[3]=(1+(theta_aff[3]-0.5)*2*self.random_s)*np.sin(alpha)
                theta_aff[4]=(1+(theta_aff[4]-0.5)*2*self.random_s)*np.cos(alpha)
                
            if self.geometric_model=='tps' or self.geometric_model=='afftps':
                theta_tps = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
                theta_tps = theta_tps+(np.random.rand(18)-0.5)*2*self.random_t_tps
            # if self.geometric_model=='affine' or self.geometric_model=='afftps':
            #     alpha = np.random.rand(1)*np.pi/3-np.pi/6
            #     theta_aff = np.random.rand(6)
            #     theta_aff[[2,5]]=theta_aff[[2,5]]-0.5
            #     theta_aff[0]=(theta_aff[0]+0.5)*np.cos(alpha)
            #     theta_aff[1]=(theta_aff[1]+0.5)*(-np.sin(alpha))
            #     theta_aff[3]=(theta_aff[3]+0.5)*np.sin(alpha)
            #     theta_aff[4]=(theta_aff[4]+0.5)*np.cos(alpha)
                
            # if self.geometric_model=='tps' or self.geometric_model=='afftps':
            #     theta_tps = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
            #     theta_tps = theta_tps+np.random.rand(18)*0.8-0.4
            if self.geometric_model=='affine':
                theta=theta_aff
            elif self.geometric_model=='tps':
                theta=theta_tps
            elif self.geometric_model=='afftps':
                theta=np.concatenate((theta_aff,theta_tps))
            
        # make arrays float tensor for subsequent processing
        image = torch.Tensor(image.astype(np.float32))
        theta = torch.Tensor(theta.astype(np.float32))
        
        # permute order of image to CHW
        image = image.transpose(1,2).transpose(0,1)
                
        # Resize image using bilinear sampling with identity affine tnf
        if image.size()[0]!=self.out_h or image.size()[1]!=self.out_w:
            image = self.affineTnf(Variable(image.unsqueeze(0),requires_grad=False)).data.squeeze(0)
                
        sample = {'image': image, 'theta': theta}
        
        if self.transform:
            sample = self.transform(sample)

        return sample