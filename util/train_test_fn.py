from __future__ import print_function, division
import torch
from model.loss import TpsGridRegularityLoss

def train_fun_strong(epoch,model,loss_fn,optimizer,dataloader,pair_generation_tnf,use_cuda=True,log_interval=50):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta,tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()[0]
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx , len(dataloader),
                100. * batch_idx / len(dataloader), loss.data[0]))
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss

def test_fun_strong(model,loss_fn,dataloader,pair_generation_tnf,use_cuda=True):
    model.eval()
    test_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta,tnf_batch['theta_GT'])
        test_loss += loss.data.cpu().numpy()[0]

    test_loss /= len(dataloader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss


def train_fun_weak(epoch,model,loss_fn,optimizer,dataloader,dataloader_neg,batch_tnf,use_cuda=True,log_interval=50,triplet=False,tps_grid_regularity_loss=0):
    tgrl = TpsGridRegularityLoss(use_cuda=use_cuda)
    model.train()
    train_loss = 0
    if dataloader_neg is not None: 
        dataloader_neg_iter=iter(dataloader_neg)    
    for batch_idx, batch in enumerate(dataloader):        
        optimizer.zero_grad()
        batch = batch_tnf(batch)
        if dataloader_neg is not None and triplet==False: 
            batch_neg = next(dataloader_neg_iter)
            batch_neg = batch_tnf(batch_neg)
            theta_pos,corr_pos,theta_neg,corr_neg = model(batch, batch_neg)
            inliers_pos = loss_fn(theta_pos,corr_pos)            
            inliers_neg = loss_fn(theta_neg,corr_neg)
            loss = torch.sum(inliers_neg - inliers_pos)
        elif dataloader_neg is None and triplet==False:
            theta,corr = model(batch)
            loss = loss_fn(theta,corr)
        elif dataloader_neg is None and triplet==True:
            f_A = model.FeatureExtraction(batch['source_image'])
            f_B = model.FeatureExtraction(batch['source_image'])
            f_N = model.FeatureExtraction(batch['negative_image'])
            corr_pos = model.FeatureCorrelation(f_A,f_B)
            corr_neg = model.FeatureCorrelation(f_A,f_N)
            theta_pos = model.FeatureRegression(corr_pos)
            theta_neg = model.FeatureRegression(corr_neg)
            inliers_pos = loss_fn(theta_pos,corr_pos)            
            inliers_neg = loss_fn(theta_neg,corr_neg)
            loss = torch.sum(inliers_neg - inliers_pos)
            if tps_grid_regularity_loss != 0:
                loss = loss + tps_grid_regularity_loss*tgrl(theta_pos)

        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()[0]
        print_train_progress(log_interval,batch_idx,len(dataloader),epoch,loss.data[0])
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def test_fun_weak(model,loss_fn,dataloader,dataloader_neg,batch_tnf,use_cuda=True,triplet=False,tps_grid_regularity_loss=0):
    model.eval()
    test_loss = 0
    if dataloader_neg is not None: 
        dataloader_neg_iter=iter(dataloader_neg)
    for batch_idx, batch in enumerate(dataloader):
        batch = batch_tnf(batch)
        if dataloader_neg is not None: 
            batch_neg = next(dataloader_neg_iter)
            batch_neg = batch_tnf(batch_neg)
            theta_pos,corr_pos,theta_neg,corr_neg = model(batch, batch_neg)
            inliers_pos = loss_fn(theta_pos,corr_pos)            
            inliers_neg = loss_fn(theta_neg,corr_neg)
            loss = torch.sum(inliers_neg - inliers_pos)
        elif dataloader_neg is None and triplet==False:
            theta,corr = model(batch)
            loss = loss_fn(theta,corr)
        elif dataloader_neg is None and triplet==True:
            theta_pos,corr_pos,theta_neg,corr_neg = model(batch, triplet=True)
            inliers_pos = loss_fn(theta_pos,corr_pos)            
            inliers_neg = loss_fn(theta_neg,corr_neg)
            loss = torch.sum(inliers_neg - inliers_pos)
        test_loss += loss.data.cpu().numpy()[0]

    test_loss /= len(dataloader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss

def print_train_progress(log_interval,batch_idx,num_batches,epoch,loss_value):
    if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                    epoch, batch_idx , num_batches,
                    100. * batch_idx / num_batches, loss_value))