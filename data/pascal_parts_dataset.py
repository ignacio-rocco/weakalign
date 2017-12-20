from __future__ import print_function, division
import os
import torch
import scipy.io as scio
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf

class PascalPartsDataset(Dataset):
    
    """
    
    Pascal Parts image pair dataset
    

    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        dataset_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(self, csv_file, dataset_path, output_size=(240,240), transform=None, dataset_size=0):

        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        if dataset_size!=0:
            dataset_size = min((dataset_size,len(self.pairs)))
            self.pairs = self.pairs.iloc[0:dataset_size,:]
        self.img_A_names = self.pairs.iloc[:,0]
        self.img_B_names = self.pairs.iloc[:,1]
        self.dataset_path = dataset_path         
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
              
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A,im_size_A = self.get_image(self.img_A_names,idx)
        image_B,im_size_B = self.get_image(self.img_B_names,idx)
        
        annot_A = self.get_annot(self.img_A_names,idx)
        annot_B = self.get_annot(self.img_B_names,idx)
        
        keypoint_A,keypoint_B,part_A,part_B = self.filter_mutual_annot(annot_A,annot_B)
        
#        keypoint_A = torch.FloatTensor(keypoint_A)
        # get pre-processed point coords
#        point_A_coords = self.get_points(self.point_A_coords,idx)
#        point_B_coords = self.get_points(self.point_B_coords,idx)
        
        # compute PCK reference length L_pck (equal to max side in image_A)
        L_pck = torch.max(im_size_A)
                
        sample = {'source_image': image_A,
                  'target_image': image_B,
                  'source_im_size': im_size_A,
                  'target_im_size': im_size_B,
                  'keypoint_A': keypoint_A,
                  'keypoint_B': keypoint_B,
                  'part_A': part_A,
                  'part_B': part_B,
                  'L_pck': L_pck}
    
#        sample = {'source_image': torch.FloatTensor(2,2),'keypoint_A': np.zeros(3)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self,img_name_list,idx):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        image = io.imread(img_name)
        
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
    
    def get_annot(self,img_name_list,idx):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        annot_name = img_name[:-4]+'_new.mat'
        annot = scio.loadmat(annot_name)
        keypoint = annot['keypts']
        keypoint_flag = annot['keypts_status']
        part = annot['part_mask']
        part_flag = annot['part_mask_status']
        return (keypoint,keypoint_flag,part,part_flag)
    
    def get_points(self,point_coords_list,idx):
        point_coords = point_coords_list[idx, :].reshape(2,10)

#        # swap X,Y coords, as the the row,col order (Y,X) is used for computations
#        point_coords = point_coords[[1,0],:]

        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords

    def filter_mutual_annot(self,annot_A,annot_B):
        keypoint_A,keypoint_flag_A,part_A,part_flag_A = annot_A
        keypoint_B,keypoint_flag_B,part_B,part_flag_B = annot_B
        # get mutual keypoints
        if keypoint_A.shape!=(0,0) and keypoint_B.shape!=(0,0):
            mutual_kp_idx = np.nonzero(keypoint_flag_A * keypoint_flag_B)[1]
            keypoint_A = keypoint_A[:,mutual_kp_idx]
            keypoint_B = keypoint_B[:,mutual_kp_idx]
        else:
            keypoint_A = np.array([])
            keypoint_B = np.array([])
        # get mutual parts
        mutual_part_idx = np.nonzero(part_flag_A * part_flag_B)[1]
        if part_A.ndim==2:
            part_A = np.expand_dims(part_A,2)
        if part_B.ndim==2:
            part_B = np.expand_dims(part_B,2)
        part_A = part_A[:,:,mutual_part_idx]
        part_B = part_B[:,:,mutual_part_idx]
        return (keypoint_A,keypoint_B,part_A,part_B)