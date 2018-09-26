"""Create a multi input Mask RCNN, loading weights from multiple models.

This script takes as input an arbitrary number of models (specified by
their checkpoints and configs), and uses them to create a single model
containing a BodyMuxer with a conv body from each of the checkpoints.
"""

import argparse
import collections
import logging
import textwrap

import torch

import _init_paths  # noqa: F401
import core.config as config_utils
import utils.net as net_utils
from core.config import cfg
from modeling.model_builder import Generalized_RCNN
from modeling.body_muxer import BodyMuxer_Concatenate
from utils.logging import setup_logging


def inflate_rpn_weights(rpn_dict, model, body_index):
    def inflate(tensor, channel_index):
        new_shape = list(tensor.shape)
        new_shape[channel_index] = model.Conv_Body.dim_out
        new_tensor = torch.zeros(*new_shape)

        # Get the channel corresponding to body_index. E.g. if we care
        # about body_index 2, and the bodies have output dimension
        # [2, 5, 9]
        # then start = 7, end = 16.
        start = sum(
            x.dim_out for x in model.Conv_Body.bodies[:body_index])
        end = model.Conv_Body.bodies[body_index].dim_out

        # Create a slice that slices from start:end in the specified
        # channel_index.
        assignment_slice = [slice(None)] * new_tensor.dim()
        assignment_slice[channel_index] = slice(start, end)
        new_tensor[assignment_slice] = tensor

        return new_tensor

    # We can't directly load the weights for the following tensors, as the
    # inputs to their corresponding modules will now be bigger than before.
    # - RPN.[FPN_]RPN_conv.weight
    # - RPN.[FPN_]RPN_conv.bias,
    # - RPN.[FPN_]RPN_cls_score.weight
    # - RPN.[FPN_]RPN_bbox_pred.weight
    # layers, as  For now, we load the weights into the subtensor
    # corresponding to the --head-weights-index, leaving the rest as
    # zero.
    if 'FPN_RPN_conv.weight' in rpn_dict:
        fpn = 'FPN_'
    elif 'RPN_conv.weight' in rpn_dict:
        fpn = ''
    else:
        raise ValueError('Unknown format of rpn_dict')
    # Shape (output_channels, input_channels, w, h)
    rpn_dict[fpn + 'RPN_conv.weight'] = inflate(
        rpn_dict[fpn + 'RPN_conv.weight'], channel_index=1)
    # Shape (input_channels, )
    rpn_dict[fpn + 'RPN_conv.bias'] = inflate(
        rpn_dict[fpn + 'RPN_conv.bias'], channel_index=0)
    # Shape (output_channels, input_channels, w, h)
    rpn_dict[fpn + 'RPN_cls_score.weight'] = inflate(
        rpn_dict[fpn + 'RPN_cls_score.weight'], channel_index=1)

    # Shape (output_channels, input_channels, w, h)
    rpn_dict[fpn + 'RPN_bbox_pred.weight'] = inflate(
        rpn_dict[fpn + 'RPN_bbox_pred.weight'], channel_index=1)
    return rpn_dict


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

            if isinstance(model.Conv_Body, BodyMuxer_Concatenate):
                # XXX HACK XXX
                # If the merging method is BodyMuxer_Concatenate, we can't
                # directly load the weights for some tensors, as the inputs to
                # their corresponding modules will now be bigger than before.
                logging.info('Inflating RPN weights for BodyMuxer_Concatenate')
                inflate_rpn_weights(children_state_dicts['RPN'], model,
                                    args.head_weights_index)
                # TODO(achald): This still isn't enough, since later stage in
                # the pipeline use the RPN's dim_out to determine their
                # convolutions, so we may have to inflate _everything_ above
                # us.

            for child, child_state_dict in children_state_dicts.items():
                model._modules[child].load_state_dict(child_state_dict)
    __import__('ipdb').set_trace()


if __name__ == "__main__":
    main()
