import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import torchvision
import functools
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################

def define_loss(type='CrossEntropyLoss',weight=None):
    if type=='CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        raise NotImplementedError('loss type [%s] is not recoginized'%type)
    return criterion

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:
#             init.normal_(m.weight.data, 1.0, gain)
#             init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type is not None:
        init_weights(net,init_type,gain=init_gain)
    return net

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_16':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_net(input_nc,output_nc,net='unet',init_type='xavier_uniform',init_gain=1.0, gpu_ids=[]):
    if net != 'unet':
        init_type=None
    if net=='unet':
        net=Unet(input_nc=input_nc,output_nc=output_nc)
    elif net=='unet11':
        net=UNet11(num_classes=output_nc,pretrained=True)
#     elif net=='unet16':
#         net=UNet16(num_classes=output_nc,pretrained=True)
#     elif net == 'albunet':
#         net=AlbuNet(num_classes=output_nc,pretrained=True)
    else:
        raise NotImplementedError('model name [%s] is not recoginized'%net)
    return init_net(net,init_type=init_type,init_gain=init_gain,gpu_ids=gpu_ids)

##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = SkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = SkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = SkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = SkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = SkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class SkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

class Unet(nn.Module):
    def __init__(self,input_nc=1,output_nc=2):
        super(Unet,self).__init__()
        num_feat=[64,128,256,512,1024]
        #unet structure
        #innermost
        unet_block=UnetSkipConnectionBlock(num_feat[3],num_feat[4],innermost=True)
        #
        unet_block=UnetSkipConnectionBlock(num_feat[2],num_feat[3],submodule=unet_block)
        unet_block=UnetSkipConnectionBlock(num_feat[1],num_feat[2],submodule=unet_block)
        unet_block=UnetSkipConnectionBlock(num_feat[0],num_feat[1],submodule=unet_block)
        #outermost
        unet_block=UnetSkipConnectionBlock(output_nc,num_feat[0],input_nc=input_nc,
                                           submodule=unet_block,outermost=True)
        self.model=unet_block
    def forward(self, input):
        return self.model(input)
class UnetSkipConnectionBlock(nn.Module):
    #define the submodule with skip connection
    #----------------------------------------#
    #/-downsampling-/submodule/-upsampling-/ #
    #----------------------------------------#
    def __init__(self,outer_nc,inner_nc,input_nc=None,
                 submodule=None,outermost=False,innermost=False):
        super(UnetSkipConnectionBlock,self).__init__()
        self.outermost=outermost
        if input_nc is None:
            input_nc=outer_nc
        down_maxpool=nn.MaxPool2d(kernel_size=2)
        conv1=self.conv_3x3(input_nc,inner_nc)
        conv2 = self.conv_3x3(inner_nc * 2, inner_nc)
        up_conv=nn.ConvTranspose2d(inner_nc,outer_nc,
                                   kernel_size=2,
                                   stride=2)
        conv_out=nn.Conv2d(inner_nc,outer_nc,
                           kernel_size=1)
        if outermost:
            down=[conv1]
            up=[conv2,conv_out]
        elif innermost:
            down = [down_maxpool, conv1]
            up=[up_conv]
        else:
            down = [down_maxpool, conv1]
            up = [conv2, up_conv]

        model=down+up if innermost else down+[submodule]+up
        self.model=nn.Sequential(*model)
    def forward(self, input):
        # print('input shape:',input.size())
        output=self.model(input)
        # print('output shape:',output.size())
        if not self.outermost:
            output=torch.cat([input,output],1)
        return output

    def conv_3x3(self,input_nc,output_nc):
        conv_block1=nn.Sequential(nn.Conv2d(input_nc,output_nc,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1),
                                 nn.BatchNorm2d(output_nc),
                                 nn.ReLU(inplace=True))
        conv_block2 = nn.Sequential(nn.Conv2d(output_nc, output_nc,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(output_nc),
                                   nn.ReLU(inplace=True))
        conv_block=nn.Sequential(conv_block1,conv_block2)
        return conv_block

    
class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_classes=2, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = ConvRelu(num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(conv5)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)

# class Interpolate(nn.Module):
#     def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
#         super(Interpolate, self).__init__()
#         self.interp = nn.functional.interpolate
#         self.size = size
#         self.mode = mode
#         self.scale_factor = scale_factor
#         self.align_corners = align_corners
        
#     def forward(self, x):
#         x = self.interp(x, size=self.size, scale_factor=self.scale_factor, 
#                         mode=self.mode, align_corners=self.align_corners)
#         return x


# class DecoderBlockV2(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
#         super(DecoderBlockV2, self).__init__()
#         self.in_channels = in_channels

#         if is_deconv:
#             """
#                 Paramaters for Deconvolution were chosen to avoid artifacts, following
#                 link https://distill.pub/2016/deconv-checkerboard/
#             """

#             self.block = nn.Sequential(
#                 ConvRelu(in_channels, middle_channels),
#                 nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
#                                    padding=1),
#                 nn.ReLU(inplace=True)
#             )
#         else:
#             self.block = nn.Sequential(
#                 Interpolate(scale_factor=2, mode='bilinear'),
#                 ConvRelu(in_channels, middle_channels),
#                 ConvRelu(middle_channels, out_channels),
#             )

#     def forward(self, x):
#         return self.block(x)


# class AlbuNet(nn.Module):
#     """
#         UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
#         Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
#         """

#     def __init__(self, num_classes=2, num_filters=32, pretrained=False, is_deconv=False):
#         """
#         :param num_classes:
#         :param num_filters:
#         :param pretrained:
#             False - no pre-trained network is used
#             True  - encoder is pre-trained with resnet34
#         :is_deconv:
#             False: bilinear interpolation is used in decoder
#             True: deconvolution is used in decoder
#         """
#         super().__init__()
#         self.num_classes = num_classes

#         self.pool = nn.MaxPool2d(2, 2)

#         self.encoder = torchvision.models.resnet34(pretrained=pretrained)

#         self.relu = nn.ReLU(inplace=True)

#         self.conv1 = nn.Sequential(self.encoder.conv1,
#                                    self.encoder.bn1,
#                                    self.encoder.relu,
#                                    self.pool)

#         self.conv2 = self.encoder.layer1

#         self.conv3 = self.encoder.layer2

#         self.conv4 = self.encoder.layer3

#         self.conv5 = self.encoder.layer4

#         self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

#         self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
#         self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
#         self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
#         self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
#         self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
#         self.dec0 = ConvRelu(num_filters, num_filters)
#         self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(conv1)
#         conv3 = self.conv3(conv2)
#         conv4 = self.conv4(conv3)
#         conv5 = self.conv5(conv4)

#         center = self.center(self.pool(conv5))

#         dec5 = self.dec5(torch.cat([center, conv5], 1))

#         dec4 = self.dec4(torch.cat([dec5, conv4], 1))
#         dec3 = self.dec3(torch.cat([dec4, conv3], 1))
#         dec2 = self.dec2(torch.cat([dec3, conv2], 1))
#         dec1 = self.dec1(dec2)
#         dec0 = self.dec0(dec1)

#         x_out = self.final(dec0)

#         return x_out
    
# class UNet16(nn.Module):
#     def __init__(self, num_classes=2, num_filters=32, pretrained=False, is_deconv=False):
#         """
#         :param num_classes:
#         :param num_filters:
#         :param pretrained:
#             False - no pre-trained network used
#             True - encoder pre-trained with VGG16
#         :is_deconv:
#             False: bilinear interpolation is used in decoder
#             True: deconvolution is used in decoder
#         """
#         super().__init__()
#         self.num_classes = num_classes

#         self.pool = nn.MaxPool2d(2, 2)

#         self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

#         self.relu = nn.ReLU(inplace=True)

#         self.conv1 = nn.Sequential(self.encoder[0],
#                                    self.relu,
#                                    self.encoder[2],
#                                    self.relu)

#         self.conv2 = nn.Sequential(self.encoder[5],
#                                    self.relu,
#                                    self.encoder[7],
#                                    self.relu)

#         self.conv3 = nn.Sequential(self.encoder[10],
#                                    self.relu,
#                                    self.encoder[12],
#                                    self.relu,
#                                    self.encoder[14],
#                                    self.relu)

#         self.conv4 = nn.Sequential(self.encoder[17],
#                                    self.relu,
#                                    self.encoder[19],
#                                    self.relu,
#                                    self.encoder[21],
#                                    self.relu)

#         self.conv5 = nn.Sequential(self.encoder[24],
#                                    self.relu,
#                                    self.encoder[26],
#                                    self.relu,
#                                    self.encoder[28],
#                                    self.relu)

#         self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

#         self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
#         self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
#         self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
#         self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
#         self.dec1 = ConvRelu(64 + num_filters, num_filters)
#         self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(self.pool(conv1))
#         conv3 = self.conv3(self.pool(conv2))
#         conv4 = self.conv4(self.pool(conv3))
#         conv5 = self.conv5(self.pool(conv4))

#         center = self.center(self.pool(conv5))

#         dec5 = self.dec5(torch.cat([center, conv5], 1))

#         dec4 = self.dec4(torch.cat([dec5, conv4], 1))
#         dec3 = self.dec3(torch.cat([dec4, conv3], 1))
#         dec2 = self.dec2(torch.cat([dec3, conv2], 1))
#         dec1 = self.dec1(torch.cat([dec2, conv1], 1))

#         x_out = self.final(dec1)

#         return x_out