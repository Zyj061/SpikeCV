from torchvision.models.vgg import make_layers, load_state_dict_from_url, model_urls, VGG, cfgs
import torch.nn as nn

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    T = kwargs.get('T')
    del(kwargs['T'])
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    model.features[0] = nn.Conv2d(in_channels=T, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    nn.init.kaiming_normal_(model.features[0].weight, mode='fan_out', nonlinearity='relu')
    if model.features[0].bias is not None:
        nn.init.constant_(model.features[0].bias, 0)
    return model

def SpikeStreamVGG11_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)

def SpikeStreamVGG13_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg13_bn', 'A', True, pretrained, progress, **kwargs)

def SpikeStreamVGG16_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16_bn', 'A', True, pretrained, progress, **kwargs)

def SpikeStreamVGG19_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19_bn', 'A', True, pretrained, progress, **kwargs)