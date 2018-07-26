from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import logging
import os
import pickle
import sys
import subprocess
from collections import defaultdict
from pprint import pformat
from six.moves import xrange

# Use a non-interactive backend
import matplotlib ; matplotlib.use('Agg')

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--save_images', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args


def _set_logging(logging_filepath):
    """Setup root logger to log to file and stdout.

    All calls to logging will log to `logging_filepath` as well as stdout.
    Also creates a file logger that only logs to , which can
    be retrieved with logging.getLogger(logging_filepath).

    Args:
        logging_filepath (str): Path to log to.
    """
    log_format = ('%(asctime)s %(filename)s:%(lineno)4d: ' '%(message)s')
    stream_date_format = '%H:%M:%S'
    file_date_format = '%m/%d %H:%M:%S'

    # Clear any previous changes to logging.
    logging.root.handlers = []
    logging.root.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logging_filepath)
    file_handler.setFormatter(
        logging.Formatter(log_format, datefmt=file_date_format))
    logging.root.addHandler(file_handler)

    # Logger that logs only to file. We could also do this with levels, but
    # this allows logging specific messages regardless of level to the file
    # only (e.g. to save the diff of the current file to the log file).
    file_logger = logging.getLogger(logging_filepath)
    file_logger.addHandler(file_handler)
    file_logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(log_format, datefmt=stream_date_format))
    logging.root.addHandler(console_handler)

    logging.info('Writing log file to %s', logging_filepath)


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    logging_path = os.path.join(args.output_dir, 'detectron-pytorch.log')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    _set_logging(logging_path)

    file_logger = logging.getLogger(logging_path)
    file_logger.info('Source code:')
    file_logger.info('===')
    with open(__file__, 'r') as f:
        file_logger.info(f.read())
    file_logger.info('===')

    logging.info('Called with args:')
    logging.info(pformat(vars(args)))

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    if args.dataset == 'coco2017objectness':
        dataset = datasets.get_coco_objectness_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    logging.info('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if args.cuda:
        maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    if args.load_detectron:
        logging.info("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()

    # XXX HACK XXX
    # Look at images one nested level within folder
    NESTED_IMAGES = True

    if args.image_dir:
        if NESTED_IMAGES:
            imglist = []
            for subdir in os.listdir(args.image_dir):
                subdir_path = os.path.join(args.image_dir, subdir)
                if os.path.isdir(subdir_path):
                    imglist += misc_utils.get_imagelist_from_dir(subdir_path)
        else:
            imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
    else:
        imglist = args.images
    num_images = len(imglist)

    for image_name in imglist:
        if NESTED_IMAGES:
            parent_name = os.path.split(os.path.split(image_name)[0])[1]
            output_dir = os.path.join(args.output_dir, parent_name)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = args.output_dir
        im = cv2.imread(image_name)
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        out_image = os.path.join(
            output_dir, '{}'.format(base_name + '.png')
        )
        out_data = os.path.join(
            output_dir, '{}'.format(base_name + '.pickle')
        )
        assert im is not None

        if ((not args.save_images or os.path.isfile(out_image))
                and os.path.isfile(out_data)):
            logging.info('Already processed {}, skipping'.format(image_name))
            continue
        timers = defaultdict(Timer)

        cls_boxes, cls_segms, cls_keyps = im_detect_all(
            maskRCNN, im, timers=timers)

        logging.info('Processing {} -> {}'.format(image_name, out_data))

        if args.save_images and not os.path.isfile(out_image):
            vis_utils.vis_one_image(
                im[:, :, ::-1],  # BGR -> RGB for visualization
                base_name,
                output_dir,
                cls_boxes,
                cls_segms,
                cls_keyps,
                dataset=dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2,
                dpi=300,
                ext='png'
            )

        if not os.path.isfile(out_data):
            with open(out_data, 'wb') as f:
                data = {
                    'boxes': cls_boxes,
                    'segmentations': cls_segms,
                    'keypoints': cls_keyps,
                }
                pickle.dump(data, f)


if __name__ == '__main__':
    main()
