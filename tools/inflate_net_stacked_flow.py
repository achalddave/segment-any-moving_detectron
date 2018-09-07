"""Replicate the channels of conv1 multiple times for stacked flow.

This script has two modes:
(1) Pretrained models: Script takes as input a pretrained model, as loaded by
    resnet_weights_helper.load_pretrained_imagenet_weights, and output a .pth
    file that can also be loaded by load_pretrained_imagenet_weights. This mode
    is activated by specifying the input model using --pretrained-model.


(2) detectron.pytorch checkpoints: Script takes as input a model trained using
    detectron.pytorch, and outputs a model that can be loaded by passing to
    --load_ckpt in train_net_step.py, test_net.py, etc. This mode is activated
    by specifying the input model using --load-ckpt.

TODO(achald): This interface is rather confusing. Maybe split it up into two
different scripts?
"""

import argparse
import collections
import logging

import numpy as np
import torch
from torch import nn

import _init_paths  # noqa: E101
import utils.net as net_utils
from core.config import cfg, cfg_from_file, assert_and_infer_cfg
from modeling.model_builder import Generalized_RCNN
from utils.logging import setup_logging


def expand_weight(weight, num_flow, method):
    if method == 'copy':
        return np.tile(weight, (1, num_flow, 1, 1)) / num_flow
    elif method == 'zero':
        new_weight = np.zeros((weight.shape[0], weight.shape[1] * num_flow,
                               weight.shape[2], weight.shape[3]))
        new_weight[:, -weight.shape[1]:, :, :] = weight
        return new_weight
    else:
        raise ValueError('Unknown method: %s' % method)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--pretrained-model', help='Load pretrained-style models.')
    parser.add_argument(
        '--load-ckpt', help='Load detectron.pytorch style models.')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=81,
        help=('Set number of classes. Only used for --load-ckpt; useful if '
              'the checkpoint outputs different number of classes than the '
              '--cfg-file'))
    parser.add_argument('--cfg-file', help='Required if --load-ckpt is True.')
    parser.add_argument('--output-inflated', required=True)
    parser.add_argument('--num-flow', type=int, default=6)
    parser.add_argument(
        '--method',
        default='copy',
        choices=['copy', 'zero'],
        help=('Method for initializing weights of new channels. '
              'copy: Copy the original weights and divide them by --num-flow. '
              'zero: Set new weights to zero.'))

    args = parser.parse_args()

    assert (args.pretrained_model is None) ^ (args.load_ckpt is None), (
        'Exactly one of --pretrained-model or --load-ckpt must be specified.')

    assert args.load_ckpt is None or args.cfg_file is not None, (
        '--cfg-file must be specified if --load-ckpt is specified.')

    if args.pretrained_model is not None:
        assert args.output_inflated.endswith('.pth'), (
            'Output file must end with ".pth", or '
            '`load_pretrained_imagenet_weights` may not load it correctly.')
    else:
        assert args.output_inflated.endswith('.pth'), (
            'Output file should with ".pth" for consistency.')

    logging_path = args.output_inflated + '.log'
    setup_logging(logging_path)
    logging.info('Args:\n%s', vars(args))

    if args.pretrained_model is not None:
        state_dict = torch.load(args.pretrained_model)
        # Shape (num_outputs, num_inputs=3, kernel_width, kernel_height)
        weight = state_dict['conv1.weight']
        new_weight = expand_weight(weight, args.num_flow, args.method)
        state_dict['conv1.weight'] = torch.from_numpy(new_weight)

        torch.save(state_dict, args.output_inflated)
    else:
        checkpoint = torch.load(args.load_ckpt)
        cfg_from_file(args.cfg_file)
        cfg.MODEL.NUM_CLASSES = args.num_classes
        assert_and_infer_cfg()
        model = Generalized_RCNN()
        net_utils.load_ckpt(model, checkpoint['model'])
        stem = model.Conv_Body.conv_body.res1  # nn.Sequential
        # List of (name, module) tuples.
        stem_modules = list(stem._modules.items())

        conv1 = model.Conv_Body.conv_body.res1.conv1  # nn.Conv2d
        assert conv1.bias is None, 'Bias not implemented yet.'

        # Shape (num_outputs, num_inputs=3, kernel_width, kernel_height)
        weight = conv1.weight

        # Replicate the first two channels `num_flow` times
        new_weight = expand_weight(weight, args.num_flow, args.method)

        new_conv = nn.Conv2d(
            in_channels=new_weight.shape[1],
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            dilation=conv1.dilation,
            groups=conv1.groups,
            bias=False)
        new_conv.weight.data = torch.from_numpy(new_weight)

        stem_modules[0] = (stem_modules[0][0], new_conv)
        model.Conv_Body.conv_body.res1 = nn.Sequential(
            collections.OrderedDict(stem_modules))
        checkpoint['model'] = model.state_dict()
        torch.save(checkpoint, args.output_inflated)


if __name__ == "__main__":
    main()
