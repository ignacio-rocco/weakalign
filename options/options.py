import argparse
from util.torch_util import str_to_bool

class ArgumentParser():
    def __init__(self,mode='train'):
        self.parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
        self.add_base_parameters()
        self.add_cnn_model_parameters()
        if mode=='train_strong':
            self.add_train_parameters()
            self.add_synth_dataset_parameters()
        if mode=='train_weak':
            self.add_train_parameters()
            self.add_weak_dataset_parameters()
            self.add_weak_loss_parameters()
        elif mode=='eval':
            self.add_eval_parameters()
        
    def add_base_parameters(self):
        base_params = self.parser.add_argument_group('base')
        # Image size
        base_params.add_argument('--image-size', type=int, default=240, help='image input size')
        # Pre-trained model file
        base_params.add_argument('--model', type=str, default='', help='Pre-trained model filename')
        base_params.add_argument('--model-aff', type=str, default='', help='Trained affine model filename')
        base_params.add_argument('--model-tps', type=str, default='', help='Trained TPS model filename')
    
    def add_synth_dataset_parameters(self):
        dataset_params = self.parser.add_argument_group('dataset')
        # Dataset parameters
        dataset_params.add_argument('--dataset-csv-path', type=str, default='', help='path to training transformation csv folder')
        dataset_params.add_argument('--dataset-image-path', type=str, default='', help='path to folder containing training images')
        # Random synth dataset parameters
        dataset_params.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=False, help='sample random transformations')
        dataset_params.add_argument('--random-t', type=float, default=0.5, help='random transformation translation')
        dataset_params.add_argument('--random-s', type=float, default=0.5, help='random transformation translation')
        dataset_params.add_argument('--random-alpha', type=float, default=1/6, help='random transformation translation')
        dataset_params.add_argument('--random-t-tps', type=float, default=0.4, help='random transformation translation')        
        
    def add_weak_dataset_parameters(self):
        dataset_params = self.parser.add_argument_group('dataset')
        # Image pair dataset parameters for train/val
        dataset_params.add_argument('--dataset-csv-path', type=str, default='training_data/pf-pascal-flip/', help='path to training transformation csv folder')
        dataset_params.add_argument('--dataset-image-path', type=str, default='datasets/proposal-flow-pascal/', help='path to folder containing training images')
        dataset_params.add_argument('--categories', nargs='+', type=int, default=0, help='indices of categories for training/eval')
        # Eval dataset parameters for early stopping
        dataset_params.add_argument('--eval-dataset', type=str, default='pf-pascal', help='Validation dataset used for early stopping')
        dataset_params.add_argument('--eval-dataset-path', type=str, default='', help='path to validation dataset used for early stopping')
        dataset_params.add_argument('--pck-alpha', type=float, default=0.1, help='pck margin factor alpha')
        dataset_params.add_argument('--eval-metric', type=str, default='pck', help='pck/distance')
        # Random synth dataset parameters
        dataset_params.add_argument('--random-crop', type=str_to_bool, nargs='?', const=True, default=True, help='use random crop augmentation')                
            
    def add_train_parameters(self):
        train_params = self.parser.add_argument_group('train')
        # Optimization parameters 
        train_params.add_argument('--lr', type=float, default=0.001, help='learning rate')
        train_params.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
        train_params.add_argument('--num-epochs', type=int, default=10, help='number of training epochs')
        train_params.add_argument('--batch-size', type=int, default=16, help='training batch size')
        train_params.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
        train_params.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
        train_params.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use MSE loss on tnf. parameters')        
        train_params.add_argument('--geometric-model', type=str, default='affine', help='geometric model to be regressed at output: affine or tps')
        # Trained model parameters
        train_params.add_argument('--result-model-fn', type=str, default='checkpoint_adam', help='trained model filename')
        train_params.add_argument('--result-model-dir', type=str, default='trained_models', help='path to trained models folder')
        # Dataset name (used for loading defaults)
        train_params.add_argument('--training-dataset', type=str, default='pascal', help='dataset to use for training')
        # Limit train/test dataset sizes
        train_params.add_argument('--train-dataset-size', type=int, default=0, help='train dataset size limit')
        train_params.add_argument('--test-dataset-size', type=int, default=0, help='test dataset size limit')
        # Parts of model to train
        train_params.add_argument('--train-fe', type=str_to_bool, nargs='?', const=True, default=True, help='Train feature extraction')
        train_params.add_argument('--train-fr', type=str_to_bool, nargs='?', const=True, default=True, help='Train feature regressor')
        train_params.add_argument('--train-bn', type=str_to_bool, nargs='?', const=True, default=True, help='train batch-norm layers')
        train_params.add_argument('--fe-finetune-params',  nargs='+', type=str, default=[''], help='String indicating the F.Ext params to finetune')
        train_params.add_argument('--update-bn-buffers', type=str_to_bool, nargs='?', const=True, default=False, help='Update batch norm running mean and std')
        
    def add_weak_loss_parameters(self):
        loss_params = self.parser.add_argument_group('weak_loss')
        # Parameters of weak loss
        loss_params.add_argument('--tps-grid-size', type=int, default=3, help='tps grid size')
        loss_params.add_argument('--tps-reg-factor', type=float, default=0.2, help='tps regularization factor')
        loss_params.add_argument('--normalize-inlier-count', type=str_to_bool, nargs='?', const=True, default=True)
        loss_params.add_argument('--dilation-filter', type=int, default=0, help='type of dilation filter: 0:no filter;1:4-neighs;2:8-neighs')
        loss_params.add_argument('--use-conv-filter', type=str_to_bool, nargs='?', const=True, default=False, help='use conv filter instead of dilation')        
        
    def add_eval_parameters(self):
        eval_params = self.parser.add_argument_group('eval')
        # Evaluation parameters
        eval_params.add_argument('--eval-dataset', type=str, default='pf', help='pf/caltech/tss')
        eval_params.add_argument('--eval-dataset-path', type=str, default='', help='Path to PF dataset')
        eval_params.add_argument('--flow-output-dir', type=str, default='results/', help='flow output dir')
        eval_params.add_argument('--pck-alpha', type=float, default=0.1, help='pck margin factor alpha')
        eval_params.add_argument('--eval-metric', type=str, default='pck', help='pck/distance')
        eval_params.add_argument('--tps-reg-factor', type=float, default=0.0, help='regularisation factor for tps tnf')
        
    def add_cnn_model_parameters(self):
        model_params = self.parser.add_argument_group('model')
        # Model parameters
        model_params.add_argument('--feature-extraction-cnn', type=str, default='vgg', help='feature extraction CNN model architecture: vgg/resnet101')
        model_params.add_argument('--feature-extraction-last-layer', type=str, default='', help='feature extraction CNN last layer')
        model_params.add_argument('--fr-feature-size', type=int, default=15, help='image input size')
        model_params.add_argument('--fr-kernel-sizes', nargs='+', type=int, default=[7,5], help='kernels sizes in feat.reg. conv layers')
        model_params.add_argument('--fr-channels', nargs='+', type=int, default=[128,64], help='channels in feat. reg. conv layers')
        
    def parse(self,arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())
        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            arg_groups[group.title]=group_dict
        return (args,arg_groups)

        