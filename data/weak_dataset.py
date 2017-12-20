from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf

class ImagePairDataset(Dataset):
    
    """
    
    Image pair dataset used for weak supervision
    

    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        training_image_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(self, csv_file, training_image_path,dataset_size=None,output_size=(240,240),transform=None,random_crop=False):
        self.random_crop=random_crop
        self.out_h, self.out_w = output_size
        self.train_data = pd.read_csv(csv_file)
        if dataset_size is not None:
            dataset_size = min((dataset_size,len(self.train_data)))
            self.train_data = self.train_data.iloc[0:dataset_size,:]
        self.img_A_names = self.train_data.iloc[:,0]
        self.img_B_names = self.train_data.iloc[:,1] 
        self.set = self.train_data.iloc[:,2].as_matrix()
        self.flip = self.train_data.iloc[:, 3].as_matrix().astype('int')
        self.training_image_path = training_image_path         
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
              
    def __len__(self):
        return len(self.img_A_names)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A,im_size_A = self.get_image(self.img_A_names,idx,self.flip[idx])
        image_B,im_size_B = self.get_image(self.img_B_names,idx,self.flip[idx])
                
        image_set = self.set[idx]
                
        sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A, 'target_im_size': im_size_B, 'set':image_set}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self,img_name_list,idx,flip):
        img_name = os.path.join(self.training_image_path, img_name_list.iloc[idx])
        image = io.imread(img_name)

        # if grayscale convert to 3-channel image 
        if image.ndim==2:
            image=np.repeat(np.expand_dims(image,2),axis=2,repeats=3)
            
        # do random crop
        if self.random_crop:
            h,w,c=image.shape
            top=np.random.randint(h/4)
            bottom=int(3*h/4+np.random.randint(h/4))
            left=np.random.randint(w/4)
            right=int(3*w/4+np.random.randint(w/4))
            image = image[top:bottom,left:right,:]
            
        # flip horizontally if needed
        if flip:
            image=np.flip(image,1)
            
        # get image size
        im_size = np.asarray(image.shape)
        
        # convert to torch Variable
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image,requires_grad=False)
        
        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)
        
        im_size = torch.Tensor(im_size.astype(np.float32))
        
        return (image, im_size)

