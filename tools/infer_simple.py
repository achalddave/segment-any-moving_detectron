from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import distutils.util
import logging
import os
import pickle
import sys
import subprocess
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat
from six.moves import xrange

# Use a non-interactive backend
import matplotlib ; matplotlib.use('Agg')

from natsort import natsorted, ns
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import tools_util
import nn as mynn
from core.config import (cfg, cfg_from_file, merge_cfg_from_cfg, cfg_from_list,
                         assert_and_infer_cfg)
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from datasets import dataset_catalog
from utils.blob import pack_sequence
from utils.detectron_weight_helper import load_detectron_weight
from utils.flow import load_flow_png
from utils.logging import setup_logging
from utils.timer import Timer
from visualize_pickle import visualize

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument('--datasets', nargs='+', help='training dataset')

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
        '--image_dirs',
        nargs='*',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help=('images to infer. Must not use with --image_dirs. If the model '
              'requires multiple input datasets, use --image_dirs instead.'))
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--recursive',
        action='store_true',
        help="Look recursively in --image_dirs for images.")
    parser.add_argument(
        '--save_images', type=distutils.util.strtobool, default=True)
    parser.add_argument(
        '--vis_image_dir',
        help=('Images to use for visualization. Useful, e.g., when inferring '
              'on one modality (flow) but visualizing on another (RGB).'))
    parser.add_argument('--vis_num_workers', default=4, type=int)

    args = parser.parse_args()

    return args


def is_image(path):
    return path.is_file and any(path.suffix == extension
                                for extension in misc_utils.IMG_EXTENSIONS)


def find_extension(path):
    for extension in misc_utils.IMG_EXTENSIONS:
        if path.with_suffix(extension).exists():
            return extension
    return None


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    output_dir = Path(args.output_dir)
    logging_path = str(output_dir / (
        'detectron-pytorch-%s.log' % datetime.now().strftime('%b%d-%H-%M-%S')))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(logging_path)

    file_logger = logging.getLogger(logging_path)
    file_logger.info('Source code:')
    file_logger.info('===')
    with open(__file__, 'r') as f:
        file_logger.info(f.read())
    file_logger.info('===')

    logging.info('Called with args:')
    logging.info(pformat(vars(args)))

    assert args.image_dirs or args.images
    assert bool(args.image_dirs) ^ bool(args.images)

    # If the config is a pickle file, the pixel means should already have been
    # edited at train time if necessary. Assume that the training code knew
    # better, and don't edit them here..
    config_is_pickle = args.cfg_file.endswith('.pkl')
    tools_util.update_cfg_for_dataset(
        args.datasets, update_pixel_means=not config_is_pickle)

    if cfg.MODEL.NUM_CLASSES == 2:
        dataset = datasets.get_objectness_dataset()
    elif cfg.MODEL.NUM_CLASSES == 81:
        dataset = datasets.get_coco_dataset()

    input_is_flow = [
        dataset_catalog.DATASETS[x][dataset_catalog.IS_FLOW]
        for x in args.datasets
    ]
    for i, is_flow in enumerate(input_is_flow):
        if is_flow:
            logging.info(f'Input {i} treated as flow images.')

    logging.info('load cfg from file: {}'.format(args.cfg_file))
    if args.cfg_file.endswith('.pkl'):
        import yaml
        with open(args.cfg_file, 'rb') as f:
            other_cfg = yaml.load(pickle.load(f)['cfg'])
        detectron_dir = Path(__file__).parent.parent
        if Path(other_cfg['ROOT_DIR']) != detectron_dir:
            other_cfg['ROOT_DIR'] = str(detectron_dir)
            logging.info(
                'Updating ROOT_DIR in loaded config to '
                'current ROOT_DIR: %s' % other_cfg['ROOT_DIR'])
        merge_cfg_from_cfg(other_cfg)
    else:
        cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    file_logger.info('Config:')
    file_logger.info(pformat(cfg))

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

    if args.image_dirs:
        image_dirs = [Path(x) for x in args.image_dirs]

        # Collect all images from each image dir.
        # all_images[i] contains a mapping from relative image paths to the
        # absolute image path for image_dirs[i].
        all_images = []
        for image_dir in image_dirs:
            if args.recursive:
                images = [x for x in image_dir.rglob('*') if is_image(x)]
            else:
                images = [x for x in image_dir.iterdir() if is_image(x)]
            if not images:
                error = f'Found no images in {image_dir}.'
                if (not args.recursive
                        and any(x.is_dir() for x in image_dir.iterdir())):
                    error += ' Did you mean to specify --recursive?'
                raise ValueError(error)
            all_images.append({
                x.relative_to(image_dir).with_suffix(''): x
                for x in images
            })

        all_relative_paths = [set(x.keys()) for x in all_images]

        # Collect intersection of relative paths from each image dir.
        relative_paths_union = set.union(*all_relative_paths)
        relative_paths = set.intersection(*all_relative_paths)
        num_missing = len(relative_paths_union) - len(relative_paths)
        if num_missing:
            logging.info(
                f'Found {len(relative_paths_union)} unique images, '
                f'but only {len(relative_paths)} are in all --image_dirs. '
                f'({num_missing} missing)')

        relative_paths = natsorted(relative_paths, alg=ns.PATH)
        # Map relative path to list of images ordered as with image_dir
        images = [
            tuple(image_map[relative_path] for image_map in all_images)
            for relative_path in relative_paths
        ]
        output_images = [(output_dir / x).with_suffix('.png')
                         for x in relative_paths]

        if args.vis_image_dir:
            vis_image_dir = Path(args.vis_image_dir)
            # Try to guess the right visualization extension.
            first_image = vis_image_dir / relative_paths[0]
            vis_extension = find_extension(first_image)
            if vis_extension is None:
                raise ValueError(
                    "Couldn't find visualization image with any extension: %s"
                    % first_image.with_suffix(''))
            vis_images = [(vis_image_dir / x).with_suffix(vis_extension)
                          for x in relative_paths]
        else:
            vis_images = [x[0] for x in images]
    else:
        images = [(Path(x),) for x in args.images]
        output_images = [output_dir / (x[0].stem + '.png') for x in images]
        vis_images = [x[0] for x in images]
    output_pickles = [x.with_suffix('.pickle') for x in output_images]

    if args.save_images:
        # This is the type of parallel code that would really benefit from
        # using the concurrent.futures API, but unfortunately, for some reason,
        # OpenCV hangs when visualizing in a process launched through a
        # futures.ProcessPoolExecutor, but works fine through a
        # multiprocessing.Pool process.
        pool = Pool(args.vis_num_workers)

        # Match 'Infer: ' prefix later.
        vis_progress = tqdm(desc='Vis  ', position=1, total=len(images))

        def visualization_callback(result):
            vis_progress.update()

        def visualization_error(error):
            logging.error('Error when visualizing:')
            logging.error(error)

    for image_paths, vis_path, out_image, out_data in zip(
            tqdm(images, desc='Infer', position=0), vis_images,
            output_images, output_pickles):
        image_list = []
        for is_flow, image_path in zip(input_is_flow, image_paths):
            if is_flow:
                im = load_flow_png(str(image_path))
            else:
                im = cv2.imread(str(image_path))
            assert im is not None
            image_list.append(im)

        if ((not args.save_images or os.path.isfile(out_image))
                and os.path.isfile(out_data)):
            file_logger.info(
                'Already processed {}, skipping'.format(image_path))
            vis_progress.update()
            continue
        timers = defaultdict(Timer)

        cls_boxes, cls_segms, cls_keyps = im_detect_all(
            maskRCNN, pack_sequence(image_list), timers=timers)

        file_logger.info('Processing {} -> {}'.format(
            image_path, out_image if args.save_images else out_data))
        data = {
            'boxes': cls_boxes,
            'segmentations': cls_segms,
            'keypoints': cls_keyps,
        }

        if not os.path.isfile(out_data):
            out_data.parent.mkdir(exist_ok=True, parents=True)
            with open(out_data, 'wb') as f:
                pickle.dump(data, f)

        def raiser(e):
            raise e

        if args.save_images and os.path.isfile(out_image):
            vis_progress.update()
        elif args.save_images:
            out_image.parent.mkdir(exist_ok=True, parents=True)
            pool.apply_async(
                visualize,
                kwds=dict(
                    image_or_path=vis_path,
                    pickle_data_or_path=data,
                    output_path=out_image,
                    dataset=dataset,
                    thresh=0.7),
                callback=visualization_callback,
                error_callback=visualization_error)

    if args.save_images:
        pool.close()
        pool.join()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception('Caught exception: %s', e)
        sys.exit(1)
