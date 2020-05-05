import time
import torch
from options.unet_options import TrainOptions
from data import CreateDataLoader
import torchvision.transforms as tr
from models import create_model
from util.visualizer import Visualizer
from util.util import post_process

if __name__ == '__main__':
    opt = TrainOptions().parse()
    if opt.no_normalize:
        transform = tr.ToTensor()
    else:
        transform=tr.Compose([tr.ToTensor(),
                              tr.Normalize(mean=opt.transform_mean,
                                           std=opt.transform_std)
                              ])
    #mix train---syn data
    if opt.train_type=='mix':
        opt.batch_size=opt.batch_size//2
        train_loader = CreateDataLoader(opt,dataroot=opt.dataroot,image_dir=opt.train_img_dir_syn,\
                                   label_dir=opt.train_label_dir_syn,record_txt=opt.train_img_list_syn,\
                                                transform=transform,is_aug=False)
        train_dataset = train_loader.load_data()
        dataset_size = len(train_loader)
        print('#Synthetic training images = %d, batchsize = %d' %(dataset_size,opt.batch_size))
        
    #train---real data
    train_loader_real = CreateDataLoader(opt,dataroot=opt.dataroot,image_dir=opt.train_img_dir_real,\
                                   label_dir=opt.train_label_dir_real,record_txt=opt.train_img_list_real,\
                                         transform=transform,is_aug=False)
    train_dataset_real = train_loader_real.load_data()
    dataset_size_real = len(train_loader_real)
    print('#Real training images = %d, batchsize = %d' %(dataset_size_real,opt.batch_size))
        
    # eval data
    if not opt.no_eval:
        num_threads = 1
        batch_size = 1
        serial_batches = True
        val_loader = CreateDataLoader(opt,batch_size,serial_batches,num_threads,dataroot=opt.dataroot,image_dir=opt.val_img_dir,\
                                   label_dir=opt.val_label_dir,record_txt=opt.val_img_list,transform=transform,is_aug=False)
        val_dataset = val_loader.load_data()
        val_dataset_size = len(val_loader)
        print('#eval images = %d' %val_dataset_size)
    
    if opt.train_type=='mix':
        train_dataset_ = iter(train_dataset_real)
    else:
        train_dataset = train_dataset_real
        dataset_size = dataset_size_real
        
    #model    
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    
    
    #training
    opt.total_epochs = min((opt.total_iters*opt.batch_size+dataset_size)//dataset_size, opt.total_epochs)
    dataset_scale = dataset_size // dataset_size_real
    opt.save_nums_each_epoch =min(max(opt.save_nums_each_epoch,dataset_scale),4)
    save_and_display_freq = (dataset_size//opt.batch_size)*opt.batch_size//opt.save_nums_each_epoch
    test_baseline=10000
    losses={}
    total_steps = 0
    print("Training begin...total epochs:%d"%opt.total_epochs)
    model.train()
    for epoch in range(opt.epoch_count, opt.total_epochs + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataset):
            if opt.train_type == 'mix':
                try:
                    data_ = next(train_dataset_)
                except StopIteration:
                    train_dataset_ = iter(train_dataset_real)
                    data_ = next(train_dataset_)
                data['image']=torch.cat((data['image'],data_['image']),0)
                data['label']=torch.cat((data['label'],data_['label']),0)
            iter_start_time = time.time()
            if total_steps % save_and_display_freq == 0:
                t_data = iter_start_time - iter_data_time
                
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if epoch_iter % save_and_display_freq==0:
                
                save_result = total_steps % (save_and_display_freq*2) == 0
                visuals = model.get_current_visuals()
                visuals.update(post_process(visuals['out'],opt.no_postprocess))
                visualizer.display_current_results(visuals, epoch, save_result,std=opt.transform_std,mean=opt.transform_mean)
                
                save_name = 'epoch_' + str(epoch) + '_' + str(i + 1)
                losses['train_loss']=model.get_current_losses()['seg']
                train_results=model.get_current_visuals()
                #eval while training
                if not opt.no_eval:
                    model.eval()
                    eval_loss=0
                    for j, data in enumerate(val_dataset):
                        model.set_input(data)
                        model.test()
                        eval_loss+=model.cal_loss()
                    model.train()
                    eval_loss/=val_dataset_size
                    losses['eval_loss']=eval_loss
                    visuals = model.get_current_visuals()
                    visuals.update(post_process(visuals['out'],opt.no_postprocess))
                    visualizer.display_current_results(visuals, epoch, False,std=opt.transform_std,mean=opt.transform_mean)
                    if eval_loss<test_baseline:
                        print('saving the eval best model (epoch %d, total_steps %d)' % (epoch, total_steps))
                        test_baseline=eval_loss
                        model.save_networks('best_eval')
                visualizer.plot_current_losses(epoch,float(epoch_iter) / dataset_size,losses)
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch,i + 1,losses, t, t_data)
            #save models
            if epoch_iter % save_and_display_freq==0 and epoch % opt.save_epoch_freq==0:
                print('saving the model (epoch_%d_%d)' % (epoch, i+1))
                save_name = 'epoch_' + str(epoch) + '_' + str(i + 1)
                model.save_networks('latest')
                model.save_networks(save_name)
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.total_epochs, time.time() - epoch_start_time))
        model.update_learning_rate()
