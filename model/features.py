import torch.nn as nn


class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(net_layers[1])
        self.net_layer_2 = nn.Sequential(net_layers[2])
        self.net_layer_3 = nn.Sequential(net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)
        return x1, x2, x3
