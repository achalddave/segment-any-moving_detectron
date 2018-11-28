"""Create new last layer for a specified number of classes."""

import argparse
import logging
import pprint
from pathlib import Path

import torch
from torch import nn

import _init_paths  # noqa: F401

import core.config as config_utils
import utils.net as net_utils
from core.config import cfg
from modeling.model_builder import Generalized_RCNN
from utils.logging import setup_logging


if __name__ == "__main__":
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--original-num-classes', required=True, type=int)
    parser.add_argument('--new-num-classes', required=True, type=int)
    parser.add_argument('--output-model', required=True)
    args = parser.parse_args()

    cfg.MODEL.NUM_CLASSES = args.original_num_classes
    config_utils.cfg_from_file(args.config)
    config_utils.assert_and_infer_cfg()

    setup_logging(args.output_model + '.log')
    logging.info('Script: %s' % Path(__file__).resolve())
    logging.info('Args: %s' % pprint.pformat(vars(args)))

    model = Generalized_RCNN()
    checkpoint = torch.load(
        args.checkpoint, map_location=lambda storage, loc: storage)
    net_utils.load_ckpt(model, checkpoint['model'])

    # Update box output linear layer
    original_box_cls = model.Box_Outs.cls_score
    assert isinstance(original_box_cls, nn.Linear)
    new_box_cls = nn.Linear(
        original_box_cls.in_features,
        args.new_num_classes,
        bias=(original_box_cls.bias is not None))
    model.Box_Outs.cls_score = new_box_cls

    original_box_pred = model.Box_Outs.bbox_pred
    assert isinstance(original_box_pred, nn.Linear)
    new_box_pred = nn.Linear(
        original_box_pred.in_features,
        4 * args.new_num_classes,
        bias=(original_box_pred.bias is not None))
    model.Box_Outs.bbox_pred = new_box_pred

    # Update mask output layer
    original_mask = model.Mask_Outs.classify
    assert isinstance(original_mask, nn.Conv2d)
    new_mask = nn.Conv2d(
        original_mask.in_channels,
        args.new_num_classes,
        kernel_size=original_mask.kernel_size,
        stride=original_mask.stride,
        padding=original_mask.padding,
        dilation=original_mask.dilation,
        groups=original_mask.groups,
        bias=(original_mask.bias is not None))
    model.Mask_Outs.classify = new_mask

    # The actual model state dict needs to be stored in a dictionary with
    # 'model' as the key for train_net_step.py, test_net.py, etc.
    output_checkpoint = {'model': model.state_dict()}
    torch.save(output_checkpoint, args.output_model)
