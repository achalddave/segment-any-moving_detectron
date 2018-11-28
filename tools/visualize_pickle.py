"""Visualize pickle files output by tools/infer_simple.py.

This can be used, for example, if tools/infer_simple.py was initially called
with `--save_images` set to False. It can also be used if the model needs to
be run on one type of input, but the visualizations make more sense on another
type of input (e.g. if the model runs on optical flow, but we want to visualize
on the raw pixels).
"""

import argparse
import collections
import logging
import pickle
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat

import cv2
import numpy as np
from tqdm import tqdm

import _init_paths  # noqa: F401
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.vis as vis_utils
from datasets import dataset_catalog


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


def subsample_by_parent_dir(paths, subsample_factor):
    """Subsample files at a specified rate by parent directory.

    >>> subsampled_paths = subsample_by_parent_dir(
    ...     [Path(x) for x in 
    ...      ['a/1.png', 'a/2.png', 'a/3.png', 'a/4.png', 'b/1.png']])
    >>> assert len(subsampled_paths) == 3
    >>> assert str(subsampled_paths[0]) == 'a/1.png'
    >>> assert str(subsampled_paths[1]) == 'a/4.png'
    >>> assert str(subsampled_paths[2]) == 'b/1.png'
    """
    if subsample_factor == 1:
        return paths

    import natsort
    endings = collections.defaultdict(lambda: 'th',
                                      {1: 'st', 2: 'nd', 3: 'rd'})
    pickles_by_dir = collections.defaultdict(list)
    for pickle_file in paths:
        pickles_by_dir[pickle_file.parent].append(pickle_file)

    num_before_subsampling = len(paths)
    paths = []
    for dir_pickles in pickles_by_dir.values():
        paths.extend(
            natsort.natsorted(dir_pickles,
                              alg=natsort.ns.PATH)[::subsample_factor])
    logging.info('Subsampling, visualizing every %s frame (%s / %s frames).' %
                 (str(subsample_factor) + endings[subsample_factor],
                  len(paths), num_before_subsampling))
    return paths


def visualize(image_or_path, pickle_data_or_path, output_path, dataset,
              thresh):
    if output_path.exists():
        return

    if isinstance(image_or_path, np.ndarray):
        im = image_or_path
    else:
        assert image_or_path.exists(), '%s does not exist' % image_or_path
        im = cv2.imread(str(image_or_path))

    if isinstance(pickle_data_or_path, dict):
        data = pickle_data_or_path
    else:
        with open(pickle_data_or_path, 'rb') as f:
            data = pickle.load(f)

    vis_utils.vis_one_image(
        im[:, :, ::-1],  # BGR -> RGB for visualization
        output_path.stem,
        output_path.parent,
        data['boxes'],
        data['segmentations'],
        data['keypoints'],
        dataset=dataset,
        box_alpha=0.0,
        show_class=False,
        thresh=thresh,
        kp_thresh=2,
        dpi=300,
        ext='png')


def visualize_unpack(args):
    return visualize(*args)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pickle-dir', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset', required=True, help='training dataset')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help="Look recursively in --pickle-dir for pickle files.")
    parser.add_argument(
        '--images-extension',
        default='.png')
    parser.add_argument('--threshold', default=0.7, type=float)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument(
        '--every-kth-frame',
        type=int,
        default=1,
        help=('Visualize every kth frame. Sort all pickle files using '
              'a natural sort that will respect frame ordering with typical '
              'file names (e.g. "frame001.png" or "001.png" etc.), and '
              'only visualize on every k\'th frame. If --recursive is '
              'specified, follow this procedure for every directory '
              'containing a .pickle file separately.'))

    args = parser.parse_args()

    if args.images_extension[0] != '.':
        args.images_extension = '.' + args.images_extension

    images_root = Path(args.images_dir)
    pickle_root = Path(args.pickle_dir)
    assert images_root.exists(), '--images-root does not exist'
    assert pickle_root.exists(), '--pickle-root does not exist'

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    _set_logging(str(output_root / 'visualization.log'))
    logging.info('Args: %s', pformat(vars(args)))

    if args.dataset not in dataset_catalog.DATASETS:
        raise ValueError("Unexpected args.dataset: %s" % args.dataset)
    dataset_info = dataset_catalog.DATASETS[args.dataset]
    if dataset_catalog.NUM_CLASSES not in dataset_info:
        raise ValueError(
            "Num classes not listed in dataset: %s" % args.dataset)

    if dataset_info[dataset_catalog.NUM_CLASSES] == 2:
        dataset = datasets.get_objectness_dataset()
    elif dataset_info[dataset_catalog.NUM_CLASSES] == 81:
        dataset = datasets.get_coco_dataset()

    if args.recursive:
        detectron_outputs = list(pickle_root.rglob('*.pickle')) + list(
            pickle_root.rglob('*.pkl'))
    else:
        detectron_outputs = list(pickle_root.glob('*.pickle')) + list(
            pickle_root.rglob('*.pkl'))

    if args.every_kth_frame != 1:
        detectron_outputs = subsample_by_parent_dir(detectron_outputs,
                                                    args.every_kth_frame)

    relative_paths = [x.relative_to(pickle_root) for x in detectron_outputs]
    images = [
        images_root / x.with_suffix(args.images_extension)
        for x in relative_paths
    ]
    outputs = [output_root / x.with_suffix('.png') for x in relative_paths]

    tasks = []
    for image_path, pickle_path, output_path in zip(images, detectron_outputs,
                                                    outputs):
        if output_path.exists():
            continue
        output_path.parent.mkdir(exist_ok=True, parents=True)
        tasks.append((image_path, pickle_path, output_path, dataset,
                      args.threshold))

    if not tasks:
        logging.info('Nothing to do! Exiting.')
        return

    if args.num_workers == 0:
        list(map(visualize_unpack, tqdm(tasks)))
    else:
        args.num_workers = min(args.num_workers, len(tasks))
        pool = Pool(args.num_workers)
        results = pool.imap_unordered(visualize_unpack, tasks)
        list(tqdm(results, total=len(tasks)))  # Show progress bar


if __name__ == "__main__":
    main()
