import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image,ImageOps 


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.inverse = not opt.no_label_inverse
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        ### input A (label maps)
        self.dir_A = os.path.join(self.dir_AB, opt.label_dir)
        self.A_paths = sorted(make_dataset(self.dir_A,opt.record_txt))
        if opt.isTrain:
            ### input B (real images)
            self.dir_B = os.path.join(self.dir_AB, opt.image_dir)
            self.B_paths = sorted(make_dataset(self.dir_B,opt.record_txt))

    def __getitem__(self, index):
        assert(self.opt.loadSize >= self.opt.fineSize)
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path).convert('RGB')
        ### inverse color : (boundary)255->0 && (crystal)0->255
        if self.inverse:
            A = ImageOps.invert(A)
        params = get_params(self.opt, A.size)
#         transform = get_transform(self.opt, params,method=Image.NEAREST)
        transform = get_transform(self.opt, params)
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc
        A = transform(A)
        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)
        B = 0
        if self.opt.isTrain:
            ### input B (real images)
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform = get_transform(self.opt, params)
            B = transform(B)
            if output_nc == 1:  # RGB to gray
                tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                B = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'A_path':A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'Pix2pixDataset'
