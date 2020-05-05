import torch
from .base_model import BaseModel
from . import networks
class UnetModel(BaseModel):
    def name(self):
        return 'UnetModel'
    
    @staticmethod
    def modify_commandline_options(parser,is_train=True):
        parser.set_defaults(net='unet')
        return parser

    def initialize(self,opt):
        BaseModel.initialize(self,opt)
        self.loss_names = ['seg']
        self.model_names = ['']
        self.visual_names = ['image','out']
        self.net=networks.define_net(input_nc=opt.input_nc,output_nc=opt.output_nc,net=opt.net,\
                                     init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        if self.isTrain:
            self.visual_names.append('label')
            self.optimizer=torch.optim.RMSprop(self.net.parameters(),lr=opt.lr,alpha=0.9,eps=1e-5)
            self.optimizers.append(self.optimizer)
        if isinstance(opt.loss_weight,list):
            weight=torch.Tensor(opt.loss_weight)
            weight=weight.to(self.device)
        self.criterion = networks.define_loss(opt.loss_type,weight)
    def set_input(self,input):#input(image,label,image_path)
        self.image=input['image'].to(self.device)
        if self.isTrain:
            self.label=input['label'].squeeze(1).type(torch.cuda.LongTensor).to(self.device)
        self.images_path = input['image_path']
    def forward(self):
        self.out=self.net(self.image)
    def cal_loss(self):
        self.loss_seg=self.criterion(self.out,self.label)
        return self.loss_seg.item()
    def backword(self):
        self.cal_loss()
        self.loss_seg.backward()
    def optimize_parameters(self):
        self.set_requires_grad(self.net,True)
        self.forward()
        self.optimizer.zero_grad()
        self.backword()
        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.net.parameters(),5)
        self.optimizer.step()
