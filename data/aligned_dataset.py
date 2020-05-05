from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import torchvision.transforms as tr
import torchvision.transforms.functional as F
import random
import os
from PIL import Image
import numpy as np

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    
    def initialize(self, opt,dataroot=None,image_dir=None,label_dir=None,record_txt=None,transform=None,is_aug=False):
        
        assert not None in [dataroot,image_dir,label_dir],\
        'dataroot:%s \nimage_dir:%s\nlabel_dir:%s'%(dataroot,image_dir,label_dir)
        
        self.transform=transform
        self.is_aug=is_aug
        self.input_nc = opt.input_nc
        image_folder=os.path.join(dataroot,image_dir)
        label_folder=os.path.join(dataroot,label_dir)
        self.image_paths = make_dataset(image_folder,record_txt)
        self.label_paths = make_dataset(label_folder,record_txt)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = getImg(img_path,self.input_nc) 
        label = getImg(self.label_paths[index])
#         params = get_params(self.opt, img.size)
#         transform = get_transform(self.opt, params)
#         img = transform(img)
#         transform = get_transform(self.opt, params,method=Image.NEAREST,normalize=False)
#         label = transfrom(label)

#         if self.is_aug:
#             img, label = rand_rotation(img, mask)
#             img, label = rand_verticalFlip(img, mask)
#             img, label = rand_horizontalFlip(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = tr.ToTensor()(img)
        label = tr.ToTensor()(label)
        return {'image':img,'label':label,'image_path':img_path}
    def __len__(self):
        return len(self.image_paths)

def rand_rotation(img, mask):
    '''
    img is PIL.Image object
    mask is PIL.Image object
    '''
    # 随机选择旋转角度
    angle = random.choice([90, 180, 270])

    img = F.rotate(img, angle, expand=True)
    mask = F.rotate(mask, angle, expand=True)
    return img, mask


def rand_verticalFlip(img, mask):
    '''
    img is PIL.Image object
    mask is PIL.Image object
    '''
    # 0.5的概率垂直翻转
    if random.random() < 0.5:
        img = F.vflip(img)
        mask = F.vflip(mask)
    return img, mask


def rand_horizontalFlip(img, mask):
    '''
    img is PIL.Image object
    mask is PIL.Image object
    '''
    # 0.5的概率垂直翻转
    if random.random() < 0.5:
        img = F.hflip(img)
        mask = F.hflip(mask)
    return img, mask

def getImg(img_path,input_nc=1):
    if input_nc == 3:
        img = Image.open(img_path).convert('RGB')
    else:
        img = Image.open(img_path).convert('L')
    return img
def check(dataset):
    print('\ncheck dataset')
    print('len: ',dataset.__len__())
    image,mask=dataset.__getitem__(0)
    print('size:',image.size(),'||','mask:',np.unique(mask))



