import torch
import torchvision.utils as vutils
from options import TrainOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import os
import numpy as np
import time
from util.util import mkdirs
class Trainer(object):
    def __init__(self,opt):
        super(Trainer, self).__init__()
        self.opt=opt
        #DATASET
        dataset=CreateDataLoader(opt)
        self.dataset = dataset.load_data()
        self.dataset_size = len(dataset)
        # BUILD MODEL
        self.model = create_model(opt)
        self.model.setup(opt)
        #Visualizer
        self.visualizer = Visualizer(opt)

        #TRAIN STEPS
        self.batchSize=opt.batchSize
        self.start_epoch = opt.start_epoch
        self.total_epochs=opt.total_epochs
        self.epoch_iter=opt.epoch_iter
        self.save_latest_freq=opt.save_latest_freq
        self.save_epoch_freq=opt.save_epoch_freq

        self.total_steps = (self.start_epoch - 1) * self.dataset_size + self.epoch_iter
        self.display_freq = opt.display_freq
        self.print_freq = opt.print_freq
        self.display_delta = self.total_steps % self.display_freq
        self.print_delta = self.total_steps % self.print_freq
        self.save_delta = self.total_steps % self.save_latest_freq

        self.gen_path = os.path.join(opt.checkpoints_dir, opt.name, 'outputs')
        mkdirs(self.gen_path)
        print('# initialized and get images = %d' % self.dataset_size)
        #     print(display_delta,print_delta,save_delta)
        # print(self.dataset.dataset.__getitem__(1))


    def train(self):
        z_fixed = torch.FloatTensor(self.batchSize, self.opt.z_num).normal_(0, 1)
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            epoch_start_time = time.time()
            if epoch != self.start_epoch:
                self.epoch_iter = self.epoch_iter % self.dataset_size
            for i, data in enumerate(self.dataset):
                if self.total_steps % self.print_freq == self.print_delta:
                    iter_start_time = time.time()
                self.total_steps += self.batchSize
                self.epoch_iter += self.batchSize

                ##############Forward##################
                self.model.set_input(data)
                #######################################################
                self.model.optimize_parameters()
                ##############Display results and errors ##############

                if self.total_steps % self.print_freq == self.print_delta:
                    loss_dict = self.model.get_current_losses()
                    errors = {k: v if not isinstance(v, int) else v for k, v in loss_dict.items()}
                    t = (time.time() - iter_start_time) / self.print_freq
                    self.visualizer.print_current_errors(epoch, self.epoch_iter, errors, t)
                    self.visualizer.plot_current_errors(errors, self.total_steps)
                ### display output images##############################
                if self.total_steps % self.display_freq == self.display_delta:
                    visuals = self.model.get_current_visuals()
                    self.visualizer.display_current_results(visuals, self.total_steps)
                ### save latest model
                if self.total_steps % self.save_latest_freq == self.save_delta:
                    print('saving the latest model (epoch %d,total_steps %d' % (epoch, self.total_steps))
                    save_suffix = 'latest'
                    self.model.save_networks(save_suffix)
                    np.savetxt(self.opt.iter_path, (epoch, self.epoch_iter), delimiter=',', fmt='%d')

                if self.epoch_iter >= self.dataset_size:
                    break

            ### save model for this epoch
            if epoch % self.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d,total_steps %d' % (epoch, self.total_steps))
                self.model.save_networks('latest')
                self.model.save_networks(epoch)
                np.savetxt(self.opt.iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
                self.model.eval()
                gen=self.model.generate(z_fixed)
                self.model.train()
                j=0
                for g in gen:
                    if g is not None:
                        path = '{}/{}_G_{}.png'.format(self.gen_path, self.total_steps,j)
                        vutils.save_image(g, path, nrow=self.batchSize, normalize=True, scale_each=False)
                        print("[*] Samples saved: {}".format(path))
                        j+=1
            ### end of epoch
            print('End of epoch %d / %d \t costing %d sec' % (epoch, opt.total_epochs, time.time() - epoch_start_time))
            self.model.update_learning_rate()

def setup(opt):

    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.total_epochs = 1
        opt.save_epoch_freq=1
        opt.save_latest_freq=5
        opt.max_dataset_size = 10
    else:
        opt.display_freq=1000
        opt.print_freq = 1000
    opt.iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(opt.iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at itertion %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0
    opt.start_epoch=start_epoch
    opt.epoch_iter=epoch_iter
    return opt
if __name__ == '__main__':
    args=TrainOptions().parse()
    print(args)
    # opt=setup(args)
    # trainer=Trainer(args)
    # trainer.train()








