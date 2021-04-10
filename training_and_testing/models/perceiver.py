import torch.nn as nn
from perceiver_pytorch import Perceiver



class Perceiver_custom(nn.Module):
    def __init__(self):
        super(Perceiver_custom, self).__init__()
        self.model = Perceiver(
            input_channels = 3,          # number of channels for each token of the input
            input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = 4,          # number of freq bands, with original value (2 * K + 1)
            max_freq = 5.,              # maximum frequency, hyperparameter depending on how fine the data is
            depth = 4,                   # depth of net
            num_latents = 64,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 128,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 4,            # number of heads for latent self attention, 8
            cross_dim_head = 32,
            latent_dim_head = 32,
            num_classes = 3,          # output number of classes
            attn_dropout = 0.4,
            ff_dropout = 0.4,
            weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
        )

    def forward(self, x):
        out = self.model(x)
        return out


