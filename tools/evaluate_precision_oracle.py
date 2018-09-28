"""Evaluate predictions using an oracle that removes false positives."""

import argparse
import collections
import json
import logging
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

import _init_paths  # noqa: F401

from utils.logging import setup_logging
from utils.io import save_object
from datasets.json_dataset_evaluator import _log_detection_eval_metrics
from datasets.task_evaluation import (_coco_eval_to_box_results,
                                      _coco_eval_to_mask_results)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--bbox-json',
        required=True,
        help=("Json containing list of bounding box annotations. Each "
              "annotation contains keys 'image_id', 'category_id', 'bbox', "
              "'score'."))
    parser.add_argument(
        '--mask-json',
        required=True,
        help=("Json containing list of segmentation annotations. Each "
              "annotation contains keys 'image_id', 'category_id', "
              "'segmentation', 'score'."))
    parser.add_argument('--annotations-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--iou-threshold', default=0.8, type=float)
    args = parser.parse_args()

    with open(args.bbox_json, 'r') as f:
        detections = json.load(f)

    with open(args.mask_json, 'r') as f:
        segmentations = json.load(f)

    groundtruth = COCO(args.annotations_json)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True)

    logging_path = str(output_root / (Path(__file__).name + '.log'))
    setup_logging(logging_path)
    logging.info('Args:\n%s' % vars(args))

    file_logger = logging.getLogger(logging_path)

    detections_by_image = collections.defaultdict(list)
    for detection in detections:
        detections_by_image[detection['image_id']].append(detection)

    segmentations_by_image = collections.defaultdict(list)
    for segmentation in segmentations:
        segmentations_by_image[segmentation['image_id']].append(segmentation)

    for image_id, groundtruth_annotations in groundtruth.imgToAnns.items():
        for annotation in groundtruth_annotations:
            annotation['segmentation'] = groundtruth.annToRLE(annotation)
        filtered_detections = []
        filtered_segmentations = []

        image_detections = detections_by_image[image_id]
        image_segmentations = segmentations_by_image[image_id]
        # (num_detections, num_groundtruth)
        mask_ious = mask_utils.iou(
            [x['segmentation'] for x in image_segmentations],
            [x['segmentation'] for x in groundtruth_annotations],
            pyiscrowd=[0 for _ in groundtruth_annotations])

        for i, (detection, segmentation) in enumerate(
                zip(image_detections, image_segmentations)):
            if np.any(mask_ious[i] > args.iou_threshold):
                filtered_detections.append(detection)
                filtered_segmentations.append(segmentation)
        file_logger.info(
            'Kept %s/%s detections' % (len(filtered_detections),
                                       len(detections_by_image[image_id])))
        detections_by_image[image_id] = filtered_detections
        segmentations_by_image[image_id] = filtered_segmentations

    detection_output_json = str(output_root / Path(args.bbox_json).name)
    segmentation_output_json = str(output_root / Path(args.mask_json).name)
    with open(detection_output_json, 'w') as f:
        detections_flat = [
            d for ds in detections_by_image.values() for d in ds
        ]
        json.dump(detections_flat, f)

    with open(segmentation_output_json, 'w') as f:
        segmentations_flat = [
            d for ds in segmentations_by_image.values() for d in ds
        ]
        json.dump(segmentations_flat, f)

    classes = [x['name'] for x in groundtruth.cats.values()]
    classes = ['__background__'] + classes

    detection_results = groundtruth.loadRes(detection_output_json)
    segmentation_results = groundtruth.loadRes(segmentation_output_json)

    box_coco_eval = COCOeval(groundtruth, detection_results, 'bbox')
    box_coco_eval.evaluate()
    box_coco_eval.accumulate()
    _log_detection_eval_metrics(classes, box_coco_eval)
    eval_file = output_root / 'detection_results.pkl'
    save_object(box_coco_eval, eval_file)
    logging.info('Wrote bbox json eval results to: {}'.format(eval_file))

    mask_coco_eval = COCOeval(groundtruth, segmentation_results, 'segm')
    mask_coco_eval.evaluate()
    mask_coco_eval.accumulate()
    _log_detection_eval_metrics(classes, mask_coco_eval)
    eval_file = output_root / 'detection_results.pkl'
    save_object(mask_coco_eval, eval_file)
    logging.info('Wrote mask json eval results to: {}'.format(eval_file))

    results = {}
    results.update(_coco_eval_to_box_results(box_coco_eval))
    results.update(_coco_eval_to_mask_results(mask_coco_eval))
    to_log = [
        ('Det mAP', '%.2f' % (100 * results['box']['AP'])),
        ('Seg mAP', '%.2f' % (100 * results['mask']['AP'])),
        ('Det @ 0.5', '%.2f' % (100 * results['box']['AP50'])),
        ('Seg @ 0.5', '%.2f' % (100 * results['mask']['AP50'])),
        ('Step', ''),
        ('Train Date', ''),
        ('Path', str(Path(logging_path).resolve())),
        ('Experiment ID', '')
    ]

    logging.info('copypaste1: %s', ','.join(x[0] for x in to_log))
    logging.info('copypaste1: %s', ','.join(str(x[1]) for x in to_log))


if __name__ == "__main__":
    main()
