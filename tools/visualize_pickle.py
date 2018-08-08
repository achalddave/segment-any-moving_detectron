"""Visualize pickle files output by tools/infer_simple.py.

This can be used, for example, if tools/infer_simple.py was initially called
with `--save_images` set to False. It can also be used if the model needs to
be run on one type of input, but the visualizations make more sense on another
type of input (e.g. if the model runs on optical flow, but we want to visualize
on the raw pixels).
"""

import argparse
import logging
import pickle
from pathlib import Path
from pprint import pformat

import cv2
from tqdm import tqdm

import _init_paths  # noqa: F401
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.vis as vis_utils


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

    if (args.dataset in ('coco_2017_train_objectness',
                         'coco_2017_val_objectness')
            or any(x in args.dataset
                   for x in ['flyingthings', 'fbms', 'davis'])):
        dataset = datasets.get_objectness_dataset()
    elif args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    if args.recursive:
        detectron_outputs = list(pickle_root.rglob('*.pickle')) + list(
            pickle_root.rglob('*.pkl'))
    else:
        detectron_outputs = list(pickle_root.glob('*.pickle')) + list(
            pickle_root.rglob('*.pkl'))

    relative_paths = [x.relative_to(pickle_root) for x in detectron_outputs]
    images = [
        images_root / x.with_suffix(args.images_extension)
        for x in relative_paths
    ]
    outputs = [output_root / x.with_suffix('.png') for x in relative_paths]

    for image_path, pickle_path, output_path in zip(
            tqdm(images), detectron_outputs, outputs):
        assert image_path.exists(), '%s does not exist' % image_path
        im = cv2.imread(str(image_path))
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        output_dir = output_path.parent
        output_dir.mkdir(exist_ok=True, parents=True)
        output_basename = output_path.stem
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            output_basename,
            output_dir,
            data['boxes'],
            data['segmentations'],
            data['keypoints'],
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2,
            dpi=300,
            ext='png'
        )


if __name__ == "__main__":
    main()
