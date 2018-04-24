import torch
import numpy as np
import os
from skimage import draw
from geotnf.transformation import GeometricTnf
from geotnf.flow import th_sampling_grid_to_np_flow, write_flo_file
import torch.nn.functional as F
from data.pf_dataset import PFDataset, PFPascalDataset
from data.caltech_dataset import CaltechDataset
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf, PointsToUnitCoords, PointsToPixelCoords
from util.py_util import create_file_path
from model.loss import WeakInlierCount, TwoStageWeakInlierCount


def compute_metric(metric,model,dataset,dataloader,batch_tnf,batch_size,two_stage=True,do_aff=False,do_tps=False,args=None):
    # Initialize stats
    N=len(dataset)
    stats={}
    # decide which results should be computed aff/tps/aff+tps
    if two_stage or do_aff:
        stats['aff']={}
    if not two_stage and do_tps:
        stats['tps']={}
    if two_stage:
        stats['aff_tps']={}
    # choose metric function and metrics to compute
    if metric=='pck':  
        metrics = ['pck']
        metric_fun = pck_metric
    if metric=='dist':
        metrics = ['dist']
        metric_fun = point_dist_metric
    elif metric=='area':
        metrics = ['intersection_over_union',
                   'label_transfer_accuracy',
                   'localization_error']
        metric_fun = area_metrics
    elif metric=='pascal_parts':
        metrics = ['intersection_over_union','pck']
        metric_fun = pascal_parts_metrics
    elif metric=='flow':
        metrics = ['flow']
        metric_fun = flow_metrics
    elif metric=='inlier_count':
        metrics = ['inlier_count']
        metric_fun = inlier_count
        model.return_correlation = True
    # initialize vector for storing results for each metric
    for key in stats.keys():
        for metric in metrics:
            stats[key][metric] = np.zeros((N,1))

    # Compute
    for i, batch in enumerate(dataloader):
        batch = batch_tnf(batch)        
        batch_start_idx=batch_size*i
        batch_end_idx=np.minimum(batch_start_idx+batch_size,N)

        model.eval()
        theta_aff=None
        theta_tps=None
        theta_aff_tps=None
        
        if two_stage:
            if model.return_correlation==False:
                theta_aff,theta_aff_tps=model(batch)
            else:
                theta_aff,theta_aff_tps,corr_aff,corr_aff_tps=model(batch)
        elif do_aff:
            theta_aff=model(batch)
            if isinstance(theta_aff,tuple):
                theta_aff=theta_aff[0]
        elif do_tps:
            theta_tps=model(batch)   
            if isinstance(theta_tps,tuple):
                theta_tps=theta_tps[0]
        
        if metric=='inlier_count':
            stats = inlier_count(batch,batch_start_idx,theta_aff,theta_tps,theta_aff_tps,corr_aff,corr_aff_tps,stats,args)
        elif metric_fun is not None:
            stats = metric_fun(batch,batch_start_idx,theta_aff,theta_tps,theta_aff_tps,stats,args)
            
        print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))

    # Print results
    if metric == 'flow':
        print('Flow files have been saved to '+args.flow_output_dir)
        return stats

    for key in stats.keys():
        print('=== Results '+key+' ===')
        for metric in metrics:
            # print per-class brakedown for PFPascal, or caltech
            if isinstance(dataset,PFPascalDataset):
                N_cat = int(np.max(dataset.category))
                for c in range(N_cat):
                    cat_idx = np.nonzero(dataset.category==c+1)[0]
                    print(dataset.category_names[c].ljust(15)+': ','{:.2%}'.format(np.mean(stats[key][metric][cat_idx])))

            # print mean value
            results=stats[key][metric]
            good_idx = np.flatnonzero((results!=-1) * ~np.isnan(results))
            print('Total: '+str(results.size))
            print('Valid: '+str(good_idx.size)) 
            filtered_results = results[good_idx]
            print(metric+':','{:.2%}'.format(np.mean(filtered_results)))

        print('\n')
        
    return stats

def pck(source_points,warped_points,L_pck,alpha=0.1):
    # compute precentage of correct keypoints
    batch_size=source_points.size(0)
    pck=torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i,:]
        p_wrp = warped_points[i,:]
        N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        correct_points = torch.le(point_distance,L_pck_mat*alpha)
        pck[i]=torch.mean(correct_points.float())
    return pck

def mean_dist(source_points,warped_points,L_pck):
    # compute precentage of correct keypoints
    batch_size=source_points.size(0)
    dist=torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i,:]
        p_wrp = warped_points[i,:]
        N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        dist[i]=torch.mean(torch.div(point_distance,L_pck_mat))
    return dist

def point_dist_metric(batch,batch_start_idx,theta_aff,theta_tps,theta_aff_tps,stats,args,use_cuda=True):
    do_aff = theta_aff is not None
    do_tps = theta_tps is not None
    do_aff_tps = theta_aff_tps is not None
    
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # Instantiate point transformer
    pt = PointTnf(use_cuda=use_cuda,
                  tps_reg_factor=args.tps_reg_factor)

    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)

    if do_aff:
        # do affine only
        warped_points_aff_norm = pt.affPointTnf(theta_aff,target_points_norm)
        warped_points_aff = PointsToPixelCoords(warped_points_aff_norm,source_im_size)

    if do_tps:
        # do tps only
        warped_points_tps_norm = pt.tpsPointTnf(theta_tps,target_points_norm)
        warped_points_tps = PointsToPixelCoords(warped_points_tps_norm,source_im_size)
        
    if do_aff_tps:
        # do tps+affine
        warped_points_aff_tps_norm = pt.tpsPointTnf(theta_aff_tps,target_points_norm)
        warped_points_aff_tps_norm = pt.affPointTnf(theta_aff,warped_points_aff_tps_norm)
        warped_points_aff_tps = PointsToPixelCoords(warped_points_aff_tps_norm,source_im_size)
    
    L_pck = batch['L_pck'].data
    
    current_batch_size=batch['source_im_size'].size(0)
    indices = range(batch_start_idx,batch_start_idx+current_batch_size)

#    import pdb; pdb.set_trace()

    if do_aff:
        dist_aff = mean_dist(source_points.data, warped_points_aff.data, L_pck)
        
    if do_tps:
        dist_tps = mean_dist(source_points.data, warped_points_tps.data, L_pck)
        
    if do_aff_tps:
        dist_aff_tps = mean_dist(source_points.data, warped_points_aff_tps.data, L_pck)
        
    if do_aff:
        stats['aff']['dist'][indices] = dist_aff.unsqueeze(1).cpu().numpy()
    if do_tps:
        stats['tps']['dist'][indices] = dist_tps.unsqueeze(1).cpu().numpy()
    if do_aff_tps:
        stats['aff_tps']['dist'][indices] = dist_aff_tps.unsqueeze(1).cpu().numpy() 
        
    return stats

def inlier_count(batch,batch_start_idx,theta_aff,theta_tps,theta_aff_tps,corr_aff,corr_aff_tps,stats,args,use_cuda=True):
    inliersComposed = TwoStageWeakInlierCount(use_cuda=torch.cuda.is_available(),dilation_filter=0,normalize_inlier_count=True)
    inliers_comp = inliersComposed(matches=corr_aff,theta_aff=theta_aff,theta_aff_tps=theta_aff_tps)
    current_batch_size=batch['source_im_size'].size(0)
    indices = range(batch_start_idx,batch_start_idx+current_batch_size)
    
    stats['aff_tps']['inlier_count'][indices] = inliers_comp.unsqueeze(1).cpu().data.numpy()

    return stats


def pck_metric(batch,batch_start_idx,theta_aff,theta_tps,theta_aff_tps,stats,args,use_cuda=True):
    alpha = args.pck_alpha
    do_aff = theta_aff is not None
    do_tps = theta_tps is not None
    do_aff_tps = theta_aff_tps is not None
    
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # Instantiate point transformer
    pt = PointTnf(use_cuda=use_cuda,
                  tps_reg_factor=args.tps_reg_factor)

    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)

    if do_aff:
        # do affine only
        warped_points_aff_norm = pt.affPointTnf(theta_aff,target_points_norm)
        warped_points_aff = PointsToPixelCoords(warped_points_aff_norm,source_im_size)

    if do_tps:
        # do tps only
        warped_points_tps_norm = pt.tpsPointTnf(theta_tps,target_points_norm)
        warped_points_tps = PointsToPixelCoords(warped_points_tps_norm,source_im_size)
        
    if do_aff_tps:
        # do tps+affine
        warped_points_aff_tps_norm = pt.tpsPointTnf(theta_aff_tps,target_points_norm)
        warped_points_aff_tps_norm = pt.affPointTnf(theta_aff,warped_points_aff_tps_norm)
        warped_points_aff_tps = PointsToPixelCoords(warped_points_aff_tps_norm,source_im_size)
    
    L_pck = batch['L_pck'].data
    
    current_batch_size=batch['source_im_size'].size(0)
    indices = range(batch_start_idx,batch_start_idx+current_batch_size)

    # import pdb; pdb.set_trace()

    if do_aff:
        pck_aff = pck(source_points.data, warped_points_aff.data, L_pck, alpha)
        
    if do_tps:
        pck_tps = pck(source_points.data, warped_points_tps.data, L_pck, alpha)
        
    if do_aff_tps:
        pck_aff_tps = pck(source_points.data, warped_points_aff_tps.data, L_pck, alpha)
        
    if do_aff:
        stats['aff']['pck'][indices] = pck_aff.unsqueeze(1).cpu().numpy()
    if do_tps:
        stats['tps']['pck'][indices] = pck_tps.unsqueeze(1).cpu().numpy()
    if do_aff_tps:
        stats['aff_tps']['pck'][indices] = pck_aff_tps.unsqueeze(1).cpu().numpy() 
        
    return stats

def pascal_parts_metrics(batch,batch_start_idx,theta_aff,theta_tps,theta_aff_tps,stats,args,use_cuda=True):
    do_aff = theta_aff is not None
    do_tps = theta_tps is not None
    do_aff_tps = theta_aff_tps is not None

    batch_size=batch['source_im_size'].size(0)
    for b in range(batch_size):
        idx = batch_start_idx+b
        h_src = int(batch['source_im_size'][b,0].data.cpu().numpy())
        w_src = int(batch['source_im_size'][b,1].data.cpu().numpy())
        h_tgt = int(batch['target_im_size'][b,0].data.cpu().numpy())
        w_tgt = int(batch['target_im_size'][b,1].data.cpu().numpy())
        
        # do pck
        if batch['keypoint_A'][b].size!=0:
            src_points = Variable(torch.FloatTensor(batch['keypoint_A'][b])).unsqueeze(0)
            tgt_points = Variable(torch.FloatTensor(batch['keypoint_B'][b])).unsqueeze(0)
            L_pck = Variable(torch.FloatTensor([batch['L_pck'][b]])).unsqueeze(1)
            if use_cuda:
                src_points=src_points.cuda()
                tgt_points=tgt_points.cuda()
                L_pck = L_pck.cuda()

            batch_b = {'source_im_size': batch['source_im_size'][b,:].unsqueeze(0),
                       'target_im_size': batch['target_im_size'][b,:].unsqueeze(0),
                       'source_points':  src_points,
                       'target_points': tgt_points,
                       'L_pck': L_pck}
            args.pck_alpha = 0.05
            stats = pck_metric(batch_b,
                               idx,
                               theta_aff[b,:].unsqueeze(0) if do_aff else None,
                               theta_tps[b,:].unsqueeze(0) if do_tps else None,
                               theta_aff_tps[b,:].unsqueeze(0) if do_aff_tps else None,
                               stats,args,use_cuda)
        else:
            if do_aff:
                stats['aff']['pck'][idx] = -1
            if do_tps:
                stats['tps']['pck'][idx] = -1
            if do_aff_tps:
                stats['aff_tps']['pck'][idx] = -1
                
        # do area
        source_mask = Variable(torch.FloatTensor(batch['part_A'][b].astype(np.float32)).unsqueeze(0).transpose(2,3).transpose(1,2))
        target_mask = Variable(torch.FloatTensor(batch['part_B'][b].astype(np.float32)).unsqueeze(0).transpose(2,3).transpose(1,2))
        
        if use_cuda:
            source_mask = source_mask.cuda()
            target_mask = target_mask.cuda()
            
        grid_aff,grid_tps,grid_aff_tps=theta_to_sampling_grid(h_tgt,w_tgt,
                                                              theta_aff[b,:] if do_aff else None,
                                                              theta_tps[b,:] if do_tps else None,
                                                              theta_aff_tps[b,:] if do_aff_tps else None,
                                                              use_cuda=use_cuda,
                                                              tps_reg_factor=args.tps_reg_factor)

          
        if do_aff:
            warped_mask_aff = F.grid_sample(source_mask, grid_aff)            
            stats['aff']['intersection_over_union'][idx] = intersection_over_union(warped_mask_aff,target_mask)   
        if do_tps:
            warped_mask_tps = F.grid_sample(source_mask, grid_tps)            
            stats['tps']['intersection_over_union'][idx] = intersection_over_union(warped_mask_tps,target_mask)
        if do_aff_tps:
            warped_mask_aff_tps = F.grid_sample(source_mask, grid_aff_tps)           
            stats['aff_tps']['intersection_over_union'][idx] = intersection_over_union(warped_mask_aff_tps,target_mask)

    return stats

def area_metrics(batch,batch_start_idx,theta_aff,theta_tps,theta_aff_tps,stats,args,use_cuda=True):
    do_aff = theta_aff is not None
    do_tps = theta_tps is not None
    do_aff_tps = theta_aff_tps is not None
    
    batch_size=batch['source_im_size'].size(0)

    pt=PointTnf(use_cuda=use_cuda)

    for b in range(batch_size):
        h_src = int(batch['source_im_size'][b,0].data.cpu().numpy())
        w_src = int(batch['source_im_size'][b,1].data.cpu().numpy())
        h_tgt = int(batch['target_im_size'][b,0].data.cpu().numpy())
        w_tgt = int(batch['target_im_size'][b,1].data.cpu().numpy())

        target_mask_np,target_mask = poly_str_to_mask(batch['target_polygon'][0][b],
                                                      batch['target_polygon'][1][b],
                                                      h_tgt,w_tgt,use_cuda=use_cuda)

        source_mask_np,source_mask = poly_str_to_mask(batch['source_polygon'][0][b],
                                                      batch['source_polygon'][1][b],
                                                      h_src,w_src,use_cuda=use_cuda)

        grid_X,grid_Y = np.meshgrid(np.linspace(-1,1,w_tgt),np.linspace(-1,1,h_tgt))
        grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3)
        grid_X = Variable(grid_X,requires_grad=False)
        grid_Y = Variable(grid_Y,requires_grad=False)

        if use_cuda:
            grid_X = grid_X.cuda()
            grid_Y = grid_Y.cuda()

        grid_X_vec = grid_X.view(1,1,-1)
        grid_Y_vec = grid_Y.view(1,1,-1)

        grid_XY_vec = torch.cat((grid_X_vec,grid_Y_vec),1)        

        def pointsToGrid (x,h_tgt=h_tgt,w_tgt=w_tgt): return x.contiguous().view(1,2,h_tgt,w_tgt).transpose(1,2).transpose(2,3)

        idx = batch_start_idx+b
        
        if do_aff:
            grid_aff = pointsToGrid(pt.affPointTnf(theta_aff[b,:].unsqueeze(0),grid_XY_vec))
            warped_mask_aff = F.grid_sample(source_mask, grid_aff)            
            flow_aff = th_sampling_grid_to_np_flow(source_grid=grid_aff,h_src=h_src,w_src=w_src)
            
            stats['aff']['intersection_over_union'][idx] = intersection_over_union(warped_mask_aff,target_mask)
            stats['aff']['label_transfer_accuracy'][idx] = label_transfer_accuracy(warped_mask_aff,target_mask)
            stats['aff']['localization_error'][idx] = localization_error(source_mask_np, target_mask_np, flow_aff)
        if do_tps:
            grid_tps = pointsToGrid(pt.tpsPointTnf(theta_tps[b,:].unsqueeze(0),grid_XY_vec))
            warped_mask_tps = F.grid_sample(source_mask, grid_tps)
            flow_tps = th_sampling_grid_to_np_flow(source_grid=grid_tps,h_src=h_src,w_src=w_src)
            
            stats['tps']['intersection_over_union'][idx] = intersection_over_union(warped_mask_tps,target_mask)
            stats['tps']['label_transfer_accuracy'][idx] = label_transfer_accuracy(warped_mask_tps,target_mask)
            stats['tps']['localization_error'][idx] = localization_error(source_mask_np, target_mask_np, flow_tps)
        if do_aff_tps:
            grid_aff_tps = pointsToGrid(pt.affPointTnf(theta_aff[b,:].unsqueeze(0),pt.tpsPointTnf(theta_aff_tps[b,:].unsqueeze(0),grid_XY_vec)))
            warped_mask_aff_tps = F.grid_sample(source_mask, grid_aff_tps)
            flow_aff_tps = th_sampling_grid_to_np_flow(source_grid=grid_aff_tps,h_src=h_src,w_src=w_src)
            
            stats['aff_tps']['intersection_over_union'][idx] = intersection_over_union(warped_mask_aff_tps,target_mask)
            stats['aff_tps']['label_transfer_accuracy'][idx] = label_transfer_accuracy(warped_mask_aff_tps,target_mask)
            stats['aff_tps']['localization_error'][idx] = localization_error(source_mask_np, target_mask_np, flow_aff_tps)
        
    return stats


def flow_metrics(batch,batch_start_idx,theta_aff,theta_tps,theta_aff_tps,stats,args,use_cuda=True):
    result_path=args.flow_output_dir
    
    do_aff = theta_aff is not None
    do_tps = theta_tps is not None
    do_aff_tps = theta_aff_tps is not None

    pt=PointTnf(use_cuda=use_cuda)
    
    batch_size=batch['source_im_size'].size(0)
    for b in range(batch_size):
        h_src = int(batch['source_im_size'][b,0].data.cpu().numpy())
        w_src = int(batch['source_im_size'][b,1].data.cpu().numpy())
        h_tgt = int(batch['target_im_size'][b,0].data.cpu().numpy())
        w_tgt = int(batch['target_im_size'][b,1].data.cpu().numpy())

        grid_X,grid_Y = np.meshgrid(np.linspace(-1,1,w_tgt),np.linspace(-1,1,h_tgt))
        grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3)
        grid_X = Variable(grid_X,requires_grad=False)
        grid_Y = Variable(grid_Y,requires_grad=False)
        if use_cuda:
            grid_X = grid_X.cuda()
            grid_Y = grid_Y.cuda()

        grid_X_vec = grid_X.view(1,1,-1)
        grid_Y_vec = grid_Y.view(1,1,-1)

        grid_XY_vec = torch.cat((grid_X_vec,grid_Y_vec),1)        

        def pointsToGrid (x,h_tgt=h_tgt,w_tgt=w_tgt): return x.contiguous().view(1,2,h_tgt,w_tgt).transpose(1,2).transpose(2,3)

        idx = batch_start_idx+b
                
        if do_aff:
            grid_aff = pointsToGrid(pt.affPointTnf(theta_aff[b,:].unsqueeze(0),grid_XY_vec))
            flow_aff = th_sampling_grid_to_np_flow(source_grid=grid_aff,h_src=h_src,w_src=w_src)
            flow_aff_path = os.path.join(result_path,'aff',batch['flow_path'][b])
            create_file_path(flow_aff_path)
            write_flo_file(flow_aff,flow_aff_path)
        if do_tps:
            grid_tps = pointsToGrid(pt.tpsPointTnf(theta_tps[b,:].unsqueeze(0),grid_XY_vec))
            flow_tps = th_sampling_grid_to_np_flow(source_grid=grid_tps,h_src=h_src,w_src=w_src)
            flow_tps_path = os.path.join(result_path,'tps',batch['flow_path'][b])
            create_file_path(flow_tps_path)
            write_flo_file(flow_tps,flow_tps_path)
        if do_aff_tps:
            grid_aff_tps = pointsToGrid(pt.affPointTnf(theta_aff[b,:].unsqueeze(0),pt.tpsPointTnf(theta_aff_tps[b,:].unsqueeze(0),grid_XY_vec)))
            flow_aff_tps = th_sampling_grid_to_np_flow(source_grid=grid_aff_tps,h_src=h_src,w_src=w_src)
            flow_aff_tps_path = os.path.join(result_path,'aff_tps',batch['flow_path'][b])
            create_file_path(flow_aff_tps_path)
            write_flo_file(flow_aff_tps,flow_aff_tps_path)

        idx = batch_start_idx+b
    return stats

def poly_to_mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def poly_str_to_mask(poly_x_str,poly_y_str,out_h,out_w,use_cuda=True):
    polygon_x = np.fromstring(poly_x_str,sep=',')
    polygon_y = np.fromstring(poly_y_str,sep=',')
    mask_np = poly_to_mask(vertex_col_coords=polygon_x,
                               vertex_row_coords=polygon_y,shape=[out_h,out_w])
    mask = Variable(torch.FloatTensor(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0))
    if use_cuda:
        mask = mask.cuda()
    return (mask_np,mask)

#def intersection_over_union(warped_mask,target_mask): 
#    return torch.sum(warped_mask.data.gt(0.5) & target_mask.data.gt(0.5))/torch.sum(warped_mask.data.gt(0.5) | target_mask.data.gt(0.5))
def intersection_over_union(warped_mask,target_mask): 
    relative_part_weight = torch.sum(torch.sum(target_mask.data.gt(0.5).float(),2,True),3,True)/torch.sum(target_mask.data.gt(0.5).float())
    part_iou = torch.sum(torch.sum((warped_mask.data.gt(0.5) & target_mask.data.gt(0.5)).float(),2,True),3,True)/torch.sum(torch.sum((warped_mask.data.gt(0.5) | target_mask.data.gt(0.5)).float(),2,True),3,True)
    weighted_iou = torch.sum(torch.mul(relative_part_weight,part_iou))
    return weighted_iou

def label_transfer_accuracy(warped_mask,target_mask): 
    return torch.mean((warped_mask.data.gt(0.5) == target_mask.data.gt(0.5)).double())

def localization_error(source_mask_np, target_mask_np, flow_np):
    h_tgt, w_tgt = target_mask_np.shape[0],target_mask_np.shape[1]
    h_src, w_src = source_mask_np.shape[0],source_mask_np.shape[1]

    # initial pixel positions x1,y1 in target image
    x1, y1 = np.meshgrid(range(1,w_tgt+1), range(1,h_tgt+1))
    # sampling pixel positions x2,y2
    x2 = x1 + flow_np[:,:,0]
    y2 = y1 + flow_np[:,:,1]

    # compute in-bound coords for each image
    in_bound = (x2 >= 1) & (x2 <= w_src) & (y2 >= 1) & (y2 <= h_src)
    row,col = np.where(in_bound)
    row_1=y1[row,col].flatten().astype(np.int)-1
    col_1=x1[row,col].flatten().astype(np.int)-1
    row_2=y2[row,col].flatten().astype(np.int)-1
    col_2=x2[row,col].flatten().astype(np.int)-1

    # compute relative positions
    target_loc_x,target_loc_y = obj_ptr(target_mask_np)
    source_loc_x,source_loc_y = obj_ptr(source_mask_np)
    x1_rel=target_loc_x[row_1,col_1]
    y1_rel=target_loc_y[row_1,col_1]
    x2_rel=source_loc_x[row_2,col_2]
    y2_rel=source_loc_y[row_2,col_2]

    # compute localization error
    loc_err = np.mean(np.abs(x1_rel-x2_rel)+np.abs(y1_rel-y2_rel))
    
    return loc_err

def obj_ptr(mask):
    # computes images of normalized coordinates around bounding box
    # kept function name from DSP code
    h,w = mask.shape[0],mask.shape[1]
    y, x = np.where(mask>0.5)
    left = np.min(x);
    right = np.max(x);
    top = np.min(y);
    bottom = np.max(y);
    fg_width = right-left + 1;
    fg_height = bottom-top + 1;
    x_image,y_image = np.meshgrid(range(1,w+1), range(1,h+1));
    x_image = (x_image - left)/fg_width;
    y_image = (y_image - top)/fg_height;
    return (x_image,y_image)

