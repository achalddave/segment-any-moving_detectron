"""Visualize JSON files output by tools/test_net.py."""

import argparse
import collections
import json
import logging
import pickle
import re
from datetime import datetime
from pathlib import Path
from pprint import pformat

import cv2
from tqdm import tqdm

import _init_paths  # noqa: F401
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.vis as vis_utils
from datasets import dataset_catalog
from utils.logging import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--eval-dir',
        type=Path,
        help=('Directory containing JSON files of the format '
              'bbox_<dataset>_results.json and '
              'segmentations_<dataset>_results.json. If this is specified, '
              'the values of --dataset, --bbox-json, and --mask-json are '
              'inferred automatically.'))
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)

    parser.add_argument(
        '--bbox-json',
        type=Path,
        help=('Detection predictions in JSON format. Required unless '
              '--eval-dir is specified.'))
    parser.add_argument(
        '--mask-json',
        type=Path,
        help=('Segmentation predictions in JSON format. Required unless '
              '--eval-dir is specified.'))
    parser.add_argument(
        '--dataset',
        help=('Dataset tested on. If multiple datasets were passed as input, '
              'any single dataset can be specified. Required unless '
              '--eval-dir is specified.'))

    parser.add_argument(
        '--images-extension',
        help='Extension for images in --images-dir',
        default='.png')
    parser.add_argument(
        '--threshold', help='Visualization threshold', default=0.7, type=float)

    args = parser.parse_args()

    launch_time_str = datetime.now().strftime('%b%d-%H-%M-%S')
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(
        str(output_root / ('visualization_%s.log' % launch_time_str)))
    logging.info('Args: %s', pformat(vars(args)))

    if args.eval_dir is None:
        assert args.bbox_json is not None
        assert args.mask_json is not None
        assert args.dataset is not None
    else:
        if args.bbox_json is None:
            bbox_files = list(args.eval_dir.glob('bbox_*_results.json'))
            if len(bbox_files) > 1:
                raise ValueError('Found multiple bbox results in --eval-dir, '
                                 'so could not infer --bbox-json')
            elif len(bbox_files) == 0:
                raise ValueError('Found no bbox results in --eval-dir')
            args.bbox_json = bbox_files[0]
            logging.info(f'Inferred bbox json path: {args.bbox_json}')
        if args.mask_json is None:
            mask_files = list(
                args.eval_dir.glob('segmentations_*_results.json'))
            if len(mask_files) > 1:
                raise ValueError('Found multiple segmentation results in '
                                 '--eval-dir, so could not infer --mask-json')
            elif len(mask_files) == 0:
                raise ValueError('Found no mask results in --eval-dir')
            args.mask_json = mask_files[0]
            logging.info(f'Inferred mask json path: {args.mask_json}')
        if args.dataset is None:
            bbox_datasets, num_subs = re.subn('^bbox_(.*)_results$', r'\1',
                                              args.bbox_json.stem)
            if num_subs != 1:
                raise ValueError('Unable to extract dataset from bbox json: '
                                 f'{args.bbox_json.name}')

            mask_datasets, num_subs = re.subn('^segmentations_(.*)_results$',
                                              r'\1', args.mask_json.stem)
            if num_subs != 1:
                print(num_subs)
                raise ValueError('Unable to extract dataset from mask json: '
                                 f'{args.mask_json.name}')

            bbox_dataset = bbox_datasets.split('+')[0]
            mask_dataset = mask_datasets.split('+')[0]
            assert bbox_dataset == mask_dataset, (
                f'Bbox dataset {bbox_dataset} does not match mask dataset '
                f'{mask_dataset}')
            args.dataset = bbox_dataset
            logging.info(f'Inferred dataset: {args.dataset}')

    if args.images_extension[0] != '.':
        args.images_extension = '.' + args.images_extension

    annotations_json = dataset_catalog.DATASETS[args.dataset][
        dataset_catalog.ANN_FN]
    images_root = args.images_dir
    assert images_root.exists(), '--images-root does not exist'
    assert args.mask_json.exists()
    assert args.bbox_json.exists()

    with open(annotations_json, 'r') as f:
        groundtruth = json.load(f)
        num_categories = len(groundtruth['categories'])
        images = groundtruth['images']

    logging.info('Final args: %s', pformat(vars(args)))

    num_classes = dataset_catalog.DATASETS[args.dataset][
        dataset_catalog.NUM_CLASSES]
    if num_classes == 2:
        dataset = datasets.get_objectness_dataset()
    elif num_classes == 81:
        dataset = datasets.get_coco_dataset()
    else:
        raise ValueError(
            'Unexpected number of classes ({}) for dataset: {}'.format(
                num_classes, args.dataset))

    bbox_annotations = collections.defaultdict(list)
    with open(args.bbox_json, 'r') as f:
        bbox_annotations_raw = json.load(f)
        for ann in bbox_annotations_raw:
            bbox_annotations[ann['image_id']].append(ann)
    mask_annotations = collections.defaultdict(list)
    with open(args.mask_json, 'r') as f:
        mask_annotations_raw = json.load(f)
        for ann in mask_annotations_raw:
            mask_annotations[ann['image_id']].append(ann)

    for i, ann in enumerate(tqdm(images)):
        image_path = (images_root / ann['file_name']).with_suffix(
            args.images_extension)
        output_path = (output_root / ann['file_name']).with_suffix('.png')
        if output_path.exists():
            continue
        boxes = bbox_annotations[ann['id']]
        segmentations = mask_annotations[ann['id']]
        cls_boxes = [[] for _ in range(num_categories + 1)]
        cls_segmentations = [[] for _ in range(num_categories + 1)]
        for box, segmentation in zip(boxes, segmentations):
            x1, y1, w, h = box['bbox']
            x2, y2 = x1 + w, y1 + h
            cls_boxes[box['category_id']].append(
                [x1, y1, x2, y2, box['score']])
            cls_segmentations[segmentation['category_id']].append(
                segmentation['segmentation'])
        if not boxes:
            cls_boxes = None
            cls_segmentations = None
        assert image_path.exists(), '%s does not exist' % image_path
        im = cv2.imread(str(image_path))
        output_dir = output_path.parent
        output_dir.mkdir(exist_ok=True, parents=True)
        output_basename = output_path.stem
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            output_basename,
            output_dir,
            cls_boxes,
            cls_segmentations,
            None,
            dataset=dataset,
            box_alpha=1.0,
            show_class=True,
            thresh=args.threshold,
            kp_thresh=2,
            dpi=300,
            ext='png')


if __name__ == "__main__":
    main()
