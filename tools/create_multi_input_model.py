"""Create a multi input Mask RCNN, loading weights from multiple models.

This script takes as input an arbitrary number of models (specified by
their checkpoints and configs), and uses them to create a single model
containing a BodyMuxer with a conv body from each of the checkpoints.
"""

import argparse
import collections
import logging
import pprint
import textwrap

import torch

import _init_paths  # noqa: F401
import core.config as config_utils
import utils.net as net_utils
from core.config import cfg
from modeling import body_muxer
from modeling.model_builder import Generalized_RCNN
from utils.logging import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--body-checkpoints',
        nargs='+',
        help=textwrap.dedent("""
            Checkpoint for each input model. The order is used to specify
            the order of bodies in the BodyMuxer, and should also match the
            order of configs in --configs."""))
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--head-weights-index',
        required=True,
        type=int,
        help=textwrap.dedent("""
            Specify which of the checkpoints' weights to use for the heads and
            other parts of the models that are not the conv_body."""))
    parser.add_argument('--output-model', required=True)
    parser.add_argument(
        '--num-classes', type=int, default=2)

    args = parser.parse_args()

    cfg.MODEL.NUM_CLASSES = args.num_classes
    config_utils.cfg_from_file(args.config)
    config_utils.assert_and_infer_cfg()

    model = Generalized_RCNN()

    setup_logging(args.output_model + '.log')

    logging.info('Args: %s' % pprint.pformat(vars(args)))
    for i, checkpoint_path in enumerate(args.body_checkpoints):
        checkpoint = torch.load(checkpoint_path)['model']
        body_state_dict = {
            key.split('.', 1)[-1]: value
            for key, value in checkpoint.items() if key.startswith('Conv_Body')
        }
        model.Conv_Body.bodies[i].load_state_dict(body_state_dict)

        if i == args.head_weights_index:
            children_state_dicts = collections.defaultdict(dict)
            for key, value in checkpoint.items():
                if key.startswith('Conv_Body'):
                    continue
                child, child_key = key.split('.', 1)
                assert child_key not in children_state_dicts[child]
                children_state_dicts[child][child_key] = value

            for child, child_state_dict in children_state_dicts.items():
                model._modules[child].load_state_dict(child_state_dict)

    if isinstance(model.Conv_Body, body_muxer.BodyMuxer_ConcatenateConv):
        logging.info(
            'Initializing BodyMuxer_ConcatenateConv.conv to select '
            'RPN features from body %s directly.' % args.head_weights_index)
        model.Conv_Body.init_conv_select_index_(args.head_weights_index)

    # The actual model state dict needs to be stored in a dictionary with
    # 'model' as the key for train_net_step.py, test_net.py, etc.
    output_checkpoint = {'model': model.state_dict()}
    torch.save(output_checkpoint, args.output_model)


if __name__ == "__main__":
    main()
