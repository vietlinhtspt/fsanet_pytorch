from .resnet import ResNet
from .fsanet import FSANet

def load_model(net_type='ResNet', **kwargs):
    if net_type == 'ResNet':
        model = ResNet(**kwargs)
    elif net_type == "FSANET":
        model = FSANet().to('cuda')
    else:
        raise NotImplementedError

    return model