import torch
import torch.nn as nn
import functools

##############################################################################
# Network building blocks
##############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError(f'Normalization layer [{norm_type}] is not found')

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net

def init_net(net, init_type='normal', init_gain=0.02, device='cpu'):
    net.to(device)
    return init_weights(net, init_type, init_gain)

def get_scheduler(optimizer, policy='lambda', n_epochs=100, n_epochs_decay=100, lr_decay_iters=50):
    if policy == 'lambda':
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + 1 - n_epochs) / float(n_epochs_decay + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                                threshold=0.01, patience=5)
    else:
        raise NotImplementedError(f'learning rate policy [{policy}] is not implemented')
    return scheduler

##############################################################################
# Generator
##############################################################################

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_type='instance',
                 use_dropout=False, n_blocks=9):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        use_bias = (norm_layer == nn.InstanceNorm2d)

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # Resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer, use_dropout, use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        block = [nn.ReflectionPad2d(1),
                 nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
                 norm_layer(dim),
                 nn.ReLU(True)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [nn.ReflectionPad2d(1),
                  nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
                  norm_layer(dim)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)

##############################################################################
# Discriminator
##############################################################################

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='instance'):
        super(NLayerDiscriminator, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        use_bias = (norm_layer == nn.InstanceNorm2d)

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                   kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                         norm_layer(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                               kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                     norm_layer(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # Output: 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

##############################################################################
# Wrapper to define models
##############################################################################

def define_G(input_nc, output_nc, ngf, netG='resnet_9blocks', norm='instance',
             use_dropout=False, init_type='normal', init_gain=0.02, device='cpu'):
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm, use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm, use_dropout, n_blocks=6)
    else:
        raise NotImplementedError(f'Generator model [{netG}] is not recognized')
    return init_net(net, init_type, init_gain, device)

def define_D(input_nc, ndf, netD='basic', n_layers_D=3,
             norm='instance', init_type='normal', init_gain=0.02, device='cpu'):
    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_type=norm)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_type=norm)
    else:
        raise NotImplementedError(f'Discriminator model [{netD}] is not recognized')
    return init_net(net, init_type, init_gain, device)
