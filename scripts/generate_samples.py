import os
from options.pix2pix_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.util import mkdir,save_image,tensor2im
import time
from PIL import ImageOps
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    
    img_dir = os.path.join(opt.results_dir, opt.name, str(opt.epoch))
    mkdir(img_dir)
    
#     img_dir = os.path.join(opt.results_dir, opt.name, '%s/%s' %(opt.epoch,'images'))
#     mkdir(img_dir)
#     label_dir = os.path.join(opt.results_dir, opt.name, '%s/%s' %(opt.epoch,'labels'))
#     mkdir(label_dir)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    if opt.eval:
        model.eval()
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break 
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        syn_img = visuals['fake_B']
#         syn_label = visuals['real_A']
        syn_img = tensor2im(syn_img,nc=opt.output_nc)
#         syn_label = tensor2im(syn_label,nc=opt.input_nc)
#         if not opt.no_label_inverse:
#             syn_label = syn_label/255
#             syn_label[syn_label==0] = 255
#             syn_label[syn_label==1] = 0
#         syn_label = syn_label.astype(np.uint8)
        img_path = model.get_file_path()[0]
        name = os.path.basename(img_path)
        save_image(syn_img,os.path.join(img_dir,name))
#         save_image(syn_label,os.path.join(label_dir,name))
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
            