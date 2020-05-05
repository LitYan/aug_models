import argparse
import matplotlib.pyplot as plt
from PIL import Image
import random
import glob
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path',type=str,required=True)
    parser.add_argument('--labels_path',type=str,required=True)
    parser.add_argument('--check_num',type=int,default=10)
    opt = parser.parse_args()
    images = sorted(glob.glob(os.path.join(opt.images_path,'*.png')))
    labels = sorted(glob.glob(os.path.join(opt.labels_path,'*.png')))
    print('images num: %d |labels num : %d'%(len(images),len(labels)))
    for i in range(opt.check_num):
        index = random.randint(0,len(images)-1)
        name = os.path.basename(images[index])
        print('get %s'%name)
        if os.path.exists(os.path.join(opt.labels_path,name)):
            image = Image.open(images[index]).convert('L')
            label = Image.open(os.path.join(opt.labels_path,name)).convert('L')
            label_numpy = np.array(label,dtype=np.uint8)
            print('image size: %d %d label size:%d %d'%(image.size[0],image.size[1],label.size[0],label.size[1]))
            print('label classes:%s'%(np.unique(label_numpy)))
            plt.figure(figsize=(30,30))
            plt.subplot(1,2,1)
            plt.imshow(image,plt.cm.gray)
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(label,plt.cm.gray)
            plt.axis('off')
            plt.show()
        else:
            print('match no label !')
    
    