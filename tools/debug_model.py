import argparse

import torch

import _init_paths  # noqa: F401

import core.config as config_utils
import utils.net as net_utils
from core.config import cfg
from modeling.model_builder import Generalized_RCNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Debug script for playing with a model. Use with "
                     "'python -i' to interactively inspect the model."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--num-classes', type=int, default=2)
    args = parser.parse_args()

    cfg.MODEL.NUM_CLASSES = args.num_classes
    config_utils.cfg_from_file(args.config)
    config_utils.assert_and_infer_cfg()

    model = Generalized_RCNN()
    checkpoint = torch.load(
        args.checkpoint, map_location=lambda storage, loc: storage)
    net_utils.load_ckpt(model, checkpoint['model'])
