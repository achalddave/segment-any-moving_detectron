import torch
from torch import nn

import _init_paths
from modeling.ResNet import ResNet50_conv4_body
from modeling.body_muxer import BodyMuxer_Average


def test_weight_map():
    x = BodyMuxer_Average([ResNet50_conv4_body, ResNet50_conv4_body],
                          [[0], [1]])
    wmap, worphan = x.detectron_weight_mapping

def test_basic():
    def conv():
        c = nn.Conv2d(3, 5, 3)
        c.weight.data[:] = 1 / c.weight.data.nelement()
        return c


    class Printer(nn.Module):
        def forward(self, x):
            print(x)


    x = BodyMuxer_Average([conv, conv, conv], [[1], [2], [0]])

    inputs = torch.zeros((2, 9, 3, 3))
    inputs[:, :3] = 0
    inputs[:, 3:6] = 1
    inputs[:, 6:9] = 2
    output = x(inputs)
    print('Output shape: %s' % (output.shape, ))
    print(dir(x))

test_weight_map()
