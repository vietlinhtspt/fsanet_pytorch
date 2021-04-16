from .resnet import ResNet
from .fsanet import FSANet
from .perceiver import Perceiver_custom

def load_model(net_type='ResNet', type_model="weight", **kwargs):
    if net_type == 'ResNet':
        model = ResNet(**kwargs)
    elif net_type == "FSANET":
        model = FSANet(type_model)
    elif net_type == "Perceiver":
        model = Perceiver_custom()
    else:
        raise NotImplementedError

    return model