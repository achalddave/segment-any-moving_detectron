"""Perform inference on one or more datasets."""

import argparse
import cv2
import logging
import os
import pprint
import subprocess
import sys
from pathlib import Path

import torch
import numpy as np

import _init_paths  # pylint: disable=unused-import
import tools_util
from core.config import (cfg, merge_cfg_from_file, merge_cfg_from_cfg,
                         merge_cfg_from_list, assert_and_infer_cfg)
from core.test_engine import run_inference
from datasets import dataset_catalog
import utils.logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--datasets',
        nargs='*',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')
    parser.add_argument(
        '--objectness_eval', action='store_true',
        help=('Collapse all predicted categories to one category. Useful for '
              'evaluating object-specific detectors on objectness datasets.'))

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    output_dir_automatically_set = False
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        output_dir_automatically_set = True
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging_path = os.path.join(args.output_dir, 'evaluation.log')
    utils.logging.setup_logging(logging_path)
    subprocess.call([
        './git-state/save_git_state.sh',
        os.path.join(args.output_dir, 'git-state')
    ])

    file_logger = logging.getLogger(logging_path)
    logger = logging.getLogger(__name__)
    if output_dir_automatically_set:
        logger.info('Automatically set output directory to %s',
                    args.output_dir)

    file_logger.info('Called with args:\n')
    file_logger.info(pprint.pformat(vars(args)))

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        if args.cfg_file.endswith('.pkl'):
            import pickle
            import yaml
            with open(args.cfg_file, 'rb') as f:
                other_cfg = yaml.load(pickle.load(f)['cfg'])
                # For some reason, the RPN_COLLECT_SCALE defaults to a float,
                # but is required to be an int by the config loading code, so
                # we update it to be an int.
                try:
                    other_cfg['FPN']['RPN_COLLECT_SCALE'] = int(
                        other_cfg['FPN']['RPN_COLLECT_SCALE'])
                except KeyError:
                    pass

                detectron_dir = Path(__file__).parent.parent
                if Path(other_cfg['ROOT_DIR']) != detectron_dir:
                    other_cfg['ROOT_DIR'] = str(detectron_dir)
                    logging.info(
                        'Updating ROOT_DIR in loaded config to '
                        'current ROOT_DIR: %s' % other_cfg['ROOT_DIR'])

                merge_cfg_from_cfg(other_cfg)
        else:
            merge_cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        print('Setting cfgs')
        merge_cfg_from_list(args.set_cfgs)

    print('Pixel means: %s' % cfg.PIXEL_MEANS)
    if not args.datasets:
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    else:
        # If the config is a pickle file, the pixel means should already have
        # been edited at train time if necessary. Assume that the training code
        # knew better, and don't edit them here..
        config_is_pickle = args.cfg_file.endswith('.pkl')
        tools_util.update_cfg_for_dataset(
            args.datasets, update_pixel_means=not config_is_pickle)
        cfg.TEST.DATASETS = (args.datasets, )

    if args.objectness_eval:
        assert cfg.MODEL.NUM_CLASSES == 2
        cfg.MODEL.NUM_CLASSES = 81

    assert_and_infer_cfg()

    file_logger.info('Testing with config:')
    file_logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    all_results = run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True)

    results = all_results['+'.join(args.datasets)]
    step = ''
    experiment_id = ''
    if args.load_ckpt is not None:
        ckpt_path = Path(args.load_ckpt)
        if 'model_step' in ckpt_path.stem:
            # Should be of format 'model_step<step>'
            try:
                step = str(int(ckpt_path.stem.split('model_step')[1]))
            except ValueError as e:
                pass

        experiment_id_path = ckpt_path.parent.parent / 'experiment_id.txt'
        if experiment_id_path.exists():
            with open(experiment_id_path, 'r') as f:
                experiment_id = f.read().strip()

    to_log = [
        ('Det mAP', '%.2f' % (100 * results['box']['AP'])),
        ('Seg mAP', '%.2f' % (100 * results['mask']['AP'])),
        ('Det @ 0.5', '%.2f' % (100 * results['box']['AP50'])),
        ('Seg @ 0.5', '%.2f' % (100 * results['mask']['AP50'])),
        ('Step', step),
        ('Train Date', ''),
        ('Path', str(Path(logging_path).resolve())),
        ('Experiment ID', experiment_id)
    ]

    logging.info('copypaste1: %s', ','.join(x[0] for x in to_log))
    logging.info('copypaste1: %s', ','.join(str(x[1]) for x in to_log))
