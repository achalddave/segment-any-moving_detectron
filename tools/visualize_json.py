"""Visualize JSON files output by tools/test_net.py."""

import argparse
import collections
import json
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
from utils.logging import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bbox-json', required=True)
    parser.add_argument('--mask-json', required=True)
    parser.add_argument('--annotations-json', required=True)
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
    assert images_root.exists(), '--images-root does not exist'
    assert Path(args.mask_json).exists()
    assert Path(args.bbox_json).exists()
    assert Path(args.annotations_json).exists()

    with open(args.annotations_json, 'r') as f:
        groundtruth = json.load(f)
        num_categories = len(groundtruth['categories'])
        images = groundtruth['images']

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_root / 'visualization.log'))
    logging.info('Args: %s', pformat(vars(args)))

    if (args.dataset in ('coco_2017_train_objectness',
                         'coco_2017_val_objectness')
            or any(x in args.dataset
                   for x in ['flyingthings', 'fbms', 'davis', 'ytvos'])):
        dataset = datasets.get_objectness_dataset()
    elif args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

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
            box_alpha=0.3,
            show_class=True,
            thresh=0.5,
            kp_thresh=2,
            dpi=300,
            ext='png')


if __name__ == "__main__":
    main()
