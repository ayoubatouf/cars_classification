import torch.nn as nn
from model.net_layers import *
from model.network_wrapper import Network_Wrapper

net_layers = nn.ModuleList(
    [
        *net_layer_0(),
        *net_layer_1(),
        *net_layer_2(),
        *net_layer_3(),
        nn.ModuleList(net_layer_4()),
        nn.ModuleList(net_layer_5()),
        nn.ModuleList(net_layer_6()),
        nn.ModuleList(net_layer_7()),
    ]
)

classifier = [
    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    nn.Linear(in_features=2048, out_features=196, bias=True),
]

num_class = 196
model = Network_Wrapper(
    net_layers=net_layers, num_class=num_class, classifier=classifier
)
