import os
import numpy as np
import torchvision.transforms as tr
from options.unet_options import TestOptions
from data import CreateDataLoader
from models import create_model 
import time
from util.util import post_process,mkdir,save_image
from util.evaluation import get_map_miou_vi_ri_ari
if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    
    if opt.no_normalize:
        transform = tr.ToTensor()
    else:
        transform=tr.Compose([tr.ToTensor(),
                              tr.Normalize(mean=opt.transform_mean,
                                           std=opt.transform_std)
                              ])
        
    data_loader = CreateDataLoader(opt,dataroot=opt.dataroot,image_dir=opt.test_img_dir,\
                                   label_dir=opt.test_label_dir,record_txt=opt.test_img_list,transform=transform,is_aug=False)
    dataset = data_loader.load_data()
    datasize = len(data_loader)
    print('#test images = %d, batchsize = %d' %(datasize,opt.batch_size))
 
    model = create_model(opt)
    model.setup(opt)
    
    img_dir = os.path.join(opt.results_dir, opt.name, '%s' % opt.epoch)
    mkdir(img_dir)
    
    eval_results={}
    count=0
    with open(img_dir+'_eval.txt','w') as log:
        now = time.strftime('%c')
        log.write('=============Evaluation (%s)=============\n' % now)
    # test with eval mode.
    model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break 
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        #post process skeletonize
        pred = post_process(visuals['out'],opt.no_postprocess)['out_ske']*255
        
        img_path = model.get_file_path()[0]
        short_path = os.path.basename(img_path)
        name = os.path.splitext(short_path)[0]
        image_name = '%s.png' % name  
        save_image(pred, os.path.join(img_dir, image_name))
        
        eval_start = time.time()
        count+=1
        mask = data['label'].squeeze().numpy().astype(np.uint8)*255
        eval_result=get_map_miou_vi_ri_ari(pred,mask,boundary=opt.boundary)
        message='%04d: %s \t'%(count,name)
        for k,v in eval_result.items():
            if k in eval_results:
                eval_results[k]+=v
            else:
                eval_results[k]=v
            message+='%s: %.5f\t'%(k,v)
            
        print(message,'cost: %.4f'%(time.time()-eval_start))
        with open(img_dir + '_eval.txt', 'a') as log:
            log.write(message+'\n')
            
    message='total %d:\n'%count
    for k,v in eval_results.items():
        message += 'm_%s: %.5f\t' % (k, v/count)
    print(message)
    with open(img_dir + '_eval.txt', 'a') as log:
        log.write(message+'\n')
