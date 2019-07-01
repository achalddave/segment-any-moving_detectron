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
from datasets.json_dataset import frame_offset
from utils.blob import pack_sequence
from utils.detectron_weight_helper import load_detectron_weight
from utils.flow import load_flow_png
from utils.logging import log_argv, setup_logging
from utils.timer import Timer
from visualize_pickle import visualize

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def get_offset_images(images, offset):
    if offset == 0:
        return images

    # Collect images for each sequence.
    images_by_parent = collections.defaultdict(list)
    for image in images:
        images_by_parent[image.parent].append(image)

    offset_map = {}  # Map original image to offset image
    for sequence, frames in images_by_parent.items():
        sorted_frames = natsorted(frames, alg=ns.PATH)
        # Pad the beginning or ending of the sequence, then compute
        # the list of frames at each offset.
        if offset > 0:
            # Pad the ending of the sequence.
            offset_frames = sorted_frames + [
                sorted_frames[-1] for _ in range(offset)
            ]
            offset_frames = offset_frames[offset:]
        elif offset < 0:
            # Pad the beginning of the sequence.
            offset_frames = [
                sorted_frames[0] for _ in range(-offset)
            ] + sorted_frames
            offset_frames = offset_frames[:offset]
        assert len(sorted_frames) == len(offset_frames)
        offset_map.update(zip(sorted_frames, offset_frames))

    # Return offset images in the same order as the original images.
    return [offset_map[image] for image in images]


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
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
    parser.add_argument('--vis_threshold', default=0.7, type=float)
    parser.add_argument('--vis_num_workers', default=4, type=int)
    parser.add_argument(
        '--vis_every_kth',
        type=int,
        default=1,
        help=('Visualize every kth frame. Sort all pickle files using a '
              'natural sort that will respect frame ordering with typical '
              'file names (e.g. "frame001.png" or "001.png" etc.), and only '
              'visualize on every k\'th frame. If --recursive is specified, '
              'follow this procedure for every directory containing a .pickle '
              'file separately.'))
    parser.add_argument('--show_box', action='store_true')
    parser.add_argument('--show_class', action='store_true')

    parser.add_argument(
        '--input_types',
        choices=['rgb', 'flow'],
        nargs='*',
        help=('Indicates whether to load the input as a plain RGB image, or '
              'as an angle/magnitude flow png. If there are multiple '
              '--image_dirs, --input_types should be of the same length.'))
    parser.add_argument(
        '--num_classes',
        type=int,
        help='If specified, update num classes to this.')

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
    launch_time_str = datetime.now().strftime('%b%d-%H-%M-%S')
    output_dir = Path(args.output_dir)
    logging_path = str(output_dir / (
        'detectron-pytorch-%s.log' % launch_time_str))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(logging_path)

    file_logger = logging.getLogger(logging_path)
    subprocess.call([
        './git-state/save_git_state.sh',
        str(output_dir / ('git-state_%s' % launch_time_str))
    ])

    log_argv(file_logger)

    logging.info('Called with args:')
    logging.info(pformat(vars(args)))

    assert args.image_dirs or args.images
    assert bool(args.image_dirs) ^ bool(args.images)

    num_inputs = len(args.image_dirs) if args.image_dirs is not None else 1

    if args.input_types is None:
        logging.info('Input type not specified, assuming RGB for all.')
        args.input_types = ['rgb'] * num_inputs

    if args.image_dirs is not None:
        assert len(args.image_dirs) == len(args.input_types)

    input_is_flow = [x == 'flow' for x in args.input_types]

    if args.num_classes is None:
        args.num_classes = cfg.MODEL.NUM_CLASSES
    elif cfg.MODEL.NUM_CLASSES == -1:
        cfg.MODEL.NUM_CLASSES = args.num_classes

    if args.num_classes == 2:
        dataset = datasets.get_objectness_dataset()
    elif args.num_classes == 81:
        dataset = datasets.get_coco_dataset()
    else:
        raise ValueError(f'Unknown number of classes: {args.num_classes}')

    # If the config is a pickle file, the pixel means should already have been
    # edited at train time if necessary. Assume that the training code knew
    # better, and don't edit them here.
    should_update_pixel_means = not args.cfg_file.endswith('.pkl')
    for i, is_flow in enumerate(input_is_flow):
        if is_flow:
            logging.info(f'Input {i} treated as flow images.')
            if should_update_pixel_means:
                logging.info(f'Setting pixel mean for input {i} to 0.')
                cfg.PIXEL_MEANS[i] = np.zeros((1, 1, 3))

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
        assert (len(args.image_dirs) == len(
            cfg.DATA_LOADER.INPUT_FRAME_OFFSETS))
        image_dirs = [Path(x) for x in args.image_dirs]

        # Collect all images from each image dir.
        # all_images[i] contains a mapping from relative image paths to the
        # absolute image path for image_dirs[i].
        all_images = []
        for d, (image_dir, offset) in enumerate(
                zip(image_dirs, cfg.DATA_LOADER.INPUT_FRAME_OFFSETS)):
            if args.recursive:
                images = [x for x in image_dir.rglob('*') if is_image(x)]
                # Handle one-level of symlinks for ease of use.
                for symlink_dir in image_dir.iterdir():
                    if symlink_dir.is_symlink() and symlink_dir.is_dir():
                        images.extend(
                            [x for x in symlink_dir.rglob('*') if is_image(x)])
            else:
                images = [x for x in image_dir.iterdir() if is_image(x)]

            if not images:
                error = f'Found no images in {image_dir}.'
                if (not args.recursive
                        and any(x.is_dir() for x in image_dir.iterdir())):
                    error += ' Did you mean to specify --recursive?'
                raise ValueError(error)

            if offset == 0:
                relative_to_absolute = {
                    x.relative_to(image_dir).with_suffix(''): x
                    for x in images
                }
            else:
                offset_images = get_offset_images(images, offset)
                relative_to_absolute = {
                    image.relative_to(image_dir).with_suffix(''): offset_image
                    for image, offset_image in zip(images, offset_images)
                }

            all_images.append(relative_to_absolute)

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
            if args.input_types[0] == 'flow':
                raise ValueError(
                    "About to visualize detections on flow images, this is "
                    "likely not what you want.")
            vis_images = [x[0] for x in images]
    else:
        assert len(cfg.DATA_LOADER.INPUT_FRAME_OFFSETS) == 0
        # TODO(achald): Implement frame offsets for --images. This should be
        # pretty easy, and should just involve adding
        #   offset_images = get_offset_images(
        #       images, cfg.DATA_LOADER.INPUT_FRAME_OFFSETS)
        # Just make sure that the output_images and vis_images are aligned with
        # the non-offset images. However, I don't have time to test this so I
        # am not implementing it right now.
        if cfg.DATA_LOADER.INPUT_FRAME_OFFSETS[0] != 0:
            raise NotImplementedError(
                "Frame offsets not implemented for --images")
        images = [(Path(x),) for x in args.images]
        output_images = [output_dir / (x[0].stem + '.png') for x in images]
        vis_images = [x[0] for x in images]

    output_pickles = [x.with_suffix('.pickle') for x in output_images]

    if args.save_images:
        if args.vis_every_kth != 1:
            from visualize_pickle import subsample_by_parent_dir
            subsampled_paths = set(
                subsample_by_parent_dir(vis_images, args.vis_every_kth))
            vis_images = [
                x if x in subsampled_paths else None for x in vis_images
            ]

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
            logging.error(error.__cause__)

    for image_paths, vis_path, out_image, out_data in zip(
            tqdm(images, desc='Infer', position=0), vis_images,
            output_images, output_pickles):
        if ((not args.save_images or out_image.exists())
                and out_data.exists()):
            file_logger.info(
                'Already processed {}, skipping'.format((image_paths, )))
            if args.save_images:
                vis_progress.update()
            continue

        image_list = []
        for is_flow, image_path in zip(input_is_flow, image_paths):
            if is_flow:
                im = load_flow_png(
                    str(image_path),
                    cfg.DATA_LOADER.FLOW.LOW_MAGNITUDE_THRESHOLD)
            else:
                im = cv2.imread(str(image_path))
            assert im is not None
            image_list.append(im)
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

        if args.save_images and (out_image.exists() or vis_path is None):
            vis_progress.update()
        elif args.save_images and vis_path is not None:
            out_image.parent.mkdir(exist_ok=True, parents=True)
            pool.apply_async(
                visualize,
                kwds=dict(
                    image_or_path=vis_path,
                    pickle_data_or_path=data,
                    output_path=out_image,
                    dataset=dataset,
                    show_box=args.show_box,
                    show_class=args.show_class,
                    thresh=args.vis_threshold),
                callback=visualization_callback,
                error_callback=visualization_error)

    if args.save_images:
        pool.close()
        pool.join()

    logging.info('Saved outputs to:\n%s', output_dir.resolve())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception('Caught exception: %s', e)
        sys.exit(1)
