import argparse
import torch
import os
import models
import data
from util import util
class BaseOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--gpu_ids',type=str,default='0,1',help='gpu ids:e.g.0 0,1,2 1,2 use -1 for CPU')
        parser.add_argument('--name', type=str, required=True, help='name of the experiment.')
        parser.add_argument('--model',type=str,default='unet',choices=['unet'],help='chooses which model to use,unet...')
        parser.add_argument('--net',type=str,default='unet',choices=['unet','unet11'],help='chooses which network to use,unet...')
        parser.add_argument('--init_type', type=str, default='xavier_uniform', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=1.0, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--checkpoints_dir',type=str,default='./model_checkpoints',help='models are saved here')
        parser.add_argument('--epoch', type=str, default='best_eval', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix')
        
        # for setting inputs
        parser.add_argument('--dataroot',default='./datasets',help='path to images')
        parser.add_argument('--max_dataset_size',type=int,default=float('inf'),help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded')
        parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_normalize', action='store_true',help='do not use normalization')
        parser.add_argument('--transform_mean', type=str, default='0.94034111',help='mean for normalization,e.g. Gray:0.94034111')
        parser.add_argument('--transform_std', type=str, default='0.12718913',help='std for normalization,e.g. Gray:0.12718913')
        parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [aligned]')
        # input/output sizes 
        parser.add_argument('--batch_size',type=int,default=1,help='input batch size')
        parser.add_argument('--input_nc',type=int,default=1,help='input image channels')
        parser.add_argument('--output_nc',type=int,default=2,help='output classes')
        
        #loss
        parser.add_argument('--loss_weight',type=str,default='1:20',help='weight of loss,e.g. 1:20')
        parser.add_argument('--loss_type', choices=['CrossEntropyLoss'], default='CrossEntropyLoss', help='loss types')
        
        #post_process
        parser.add_argument('--no_postprocess', action='store_true', help='do not use postprocessing of the output ')
        #for display
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self.initialized = True
        return parser
    
    def gather_options(self):
    # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()
    
    def parse(self):
        
        opt=self.gather_options()
        opt.isTrain = self.isTrain  # train or test
        if opt.net in ['unet11','unet16','albunet']:
            opt.input_nc = 3
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
        
        #set transform mean&std
        opt.transform_mean=list(map(lambda mean:float(mean),opt.transform_mean.split(',')))
        opt.transform_std=list(map(lambda s:float(s),opt.transform_std.split(',')))
        if opt.input_nc == 3:
            opt.transform_mean*=3
            opt.transform_std*=3
        
        #set loss weight
        if opt.loss_weight != '':
            opt.loss_weight=[int(w) for w in opt.loss_weight.split(':')]
        else:
            opt.loss_weight=None
            
        #set gpu ids
        gpu_ids=opt.gpu_ids.split(',')
        opt.gpu_ids=[]
        for str_id in gpu_ids:
            id=int(str_id)
            if id >=0:
                opt.gpu_ids.append(id)
        #set cuda device
        if len(opt.gpu_ids)>0:
            torch.cuda.set_device(opt.gpu_ids[0])
            
        self.opt = opt 
        self.print_options(opt)
        return opt
    
    
    def print_options(self,opt):
        message=''
        message+='---------Options---------\n'
        for k,v in sorted(vars(opt).items()):
            comment=''
            default= self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        #save to the disk
        if self.isTrain:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
                
    
    
class TrainOptions(BaseOptions):
    def initialize(self,parser):
        parser = BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name e.g. main wxy')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for visdom')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_server', type=str, default='http://localhost', help='visdom server of the web display')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        # for training
        parser.add_argument('--train_type', type=str,default='mix',choices=['origin','mix'],
                            help='origin;use <train_img_dir>  | mix :use <train_img_dir> & <train_img_dir_syn>')
        parser.add_argument('--train_img_dir_syn', type=str, default='train/syn_images', help='where are the training images')
        parser.add_argument('--train_label_dir_syn', type=str, default='train/syn_labels', help='where are the training labels')
        parser.add_argument('--train_img_list_syn', type=str, help='train images name list')
        parser.add_argument('--train_img_dir_real',type=str, default='train/real_images', help='where are the training images')
        parser.add_argument('--train_label_dir_real',type=str, default='train/real_labels',help='where are the training labels')
        parser.add_argument('--train_img_list_real', type=str, default='./datasets/train/real_35.txt', help='train images name list')

        parser.add_argument('--no_eval',action='store_true' , help='no eval in training')
        parser.add_argument('--val_img_dir', type=str,default='val/real_images', help='where are the val images')
        parser.add_argument('--val_label_dir', type=str,default='val/real_labels', help='where are the val labels')
        parser.add_argument('--val_img_list', type=str, help='val images name list')
        
        parser.add_argument('--lr', type=float, default=1e-4, help='initial lr')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lr_gamma', type=float, default=0.8, help='multiply by a gamma every lr_decay_iters iterations')
        
        parser.add_argument('--continue_train',action='store_true',help='continue training:load which model by load_checkpoint')
        parser.add_argument('--total_iters', type=int,default=35000, help='min(total_iters*batch_size,total_epochs*datasize)|total iters to run')
        parser.add_argument('--total_epochs', type=int,default=100, help='total epochs to run')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq',type=int,default=1,help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_nums_each_epoch', type=int,default=1,help='num of saving checkpoints in each epoch')

        self.isTrain=True
        return parser


class TestOptions(BaseOptions):
    def initialize(self,parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--num_test', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--test_img_dir', type=str, default='test/real_images', help='where are the test images')
        parser.add_argument('--test_label_dir', type=str,default='test/real_labels', help='where are the test labels')
        parser.add_argument('--test_img_list', type=str,help='test images name list')
        parser.add_argument('--results_dir', type=str, default='./results/', help='save results here')
        parser.add_argument('--boundary', type=int, default=255, help='boundary mask 255|0')
        self.isTrain = False
        return parser
