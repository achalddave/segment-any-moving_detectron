"""List videos containing false positives.

Originally written to find videos in the YTVOS all-moving-strict-8-21-18 subset
that has objects which are detected by an object detector but are not labeled.
"""
import argparse
import logging
import pickle
import os
import sys
from pathlib import Path

import numpy as np

import _init_paths  # noqa: F401
from utils.logging import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--coco-eval-pickle',
        help=("Pickle file containing a COCOEval object; usually "
              "'detection_results.pkl', or 'segmentation_results.pkl'."),
        type=Path,
        required=True)
    parser.add_argument('--output-log', required=True, type=Path)
    parser.add_argument('--score-threshold', type=float, default=0.7)
    parser.add_argument('--iou-threshold', type=float, default=0.5)

    args = parser.parse_args()
    assert args.coco_eval_pickle.exists()

    setup_logging(str(args.output_log))
    logging.info('sys.argv: %s', ' '.join(sys.argv))
    logging.info('Parsed args:\n%s', vars(args))

    with open(args.coco_eval_pickle, 'rb') as f:
        coco_eval = pickle.load(f)

    images_with_unmatched_detections = set()
    images_with_missed_groundtruth = set()
    iou_matches = np.where(
        np.isclose(
            coco_eval.params.iouThrs, args.iou_threshold, rtol=0,
            atol=1e-5))[0]
    if iou_matches.size != 1:
        raise ValueError(
            'Could not find --iou-threshold (%s) in coco eval iouThrs (%s)' %
            (args.iou_threshold, coco_eval.params.iouThrs))
    iou_index = iou_matches.item()

    for eval_info in coco_eval.evalImgs:
        # Contains keys
        # ['image_id', 'category_id', 'aRng', 'maxDet', 'dtIds', 'gtIds',
        #  'dtMatches', 'gtMatches', 'dtScores', 'gtIgnore', 'dtIgnore']
        if eval_info is None:  # No detections, no groundtruth.
            continue
        image_id = eval_info['image_id']
        if image_id in images_with_unmatched_detections:
            continue

        # detection_to_groundtruth[d] contains the id of the groundtruth
        # matched to detection d, or '0' if there is no match.
        detection_to_groundtruth = (eval_info['dtMatches'][iou_index].tolist())
        detection_scores = eval_info['dtScores']
        for detection_score, matched_groundtruth in zip(
                detection_scores, detection_to_groundtruth):
            if (detection_score > args.score_threshold
                    and matched_groundtruth == 0):
                images_with_unmatched_detections.add(image_id)

        detection_id_to_index = {
            detection_id: index
            for index, detection_id in enumerate(eval_info['dtIds'])
        }
        # groundtruth_to_detection[g] contains the id of the detection
        # matched to groundtruth g, or 0 if there is no match.
        groundtruth_to_detection = eval_info['gtMatches'][iou_index].tolist()
        groundtruth_ids = eval_info['gtIds']
        for groundtruth_id, detection_match in zip(groundtruth_ids,
                                                   groundtruth_to_detection):
            assert detection_match.is_integer()
            if detection_match != 0:
                detection_score = detection_scores[detection_id_to_index[int(
                    detection_match)]]
            if (detection_match == 0
                    or detection_score < args.score_threshold):
                images_with_missed_groundtruth.add(image_id)

    sequences_with_unmatched_detections = set(
        Path(coco_eval.cocoGt.imgs[image_id]['file_name']).parent.name
        for image_id in images_with_unmatched_detections)
    sequences_with_missed_groundtruth = set(
        Path(coco_eval.cocoGt.imgs[image_id]['file_name']).parent.name
        for image_id in images_with_missed_groundtruth)
    images_with_no_mistakes = (
        set(coco_eval.cocoGt.imgs.keys()) - sequences_with_unmatched_detections
        - sequences_with_missed_groundtruth)
    all_sequences = set(
        Path(coco_eval.cocoGt.imgs[image_id]['file_name']).parent.name
        for image_id in coco_eval.cocoGt.imgs.keys())
    sequences_with_no_mistakes = (
        all_sequences - sequences_with_missed_groundtruth -
        sequences_with_unmatched_detections)

    logging.info('Num images with unmatched detections: %s',
                 len(images_with_unmatched_detections))
    logging.info('Num sequences with unmatched detections: %s',
                 len(sequences_with_unmatched_detections))
    logging.info('Sequences with unmatched detections: %s',
                 ', '.join(sorted(sequences_with_unmatched_detections)))

    logging.info('Num images with missed groundtruth: %s',
                 len(images_with_missed_groundtruth))
    logging.info('Num sequences with missed groundtruth: %s',
                 len(sequences_with_missed_groundtruth))
    logging.info('Sequences with missed groundtruth: %s',
                 ', '.join(sorted(sequences_with_missed_groundtruth)))

    logging.info('Num images with no mistakes: %s',
                 len(images_with_no_mistakes))
    logging.info('Num sequences with no mistakes: %s',
                 len(sequences_with_no_mistakes))
    logging.info('Sequences with no mistakes: %s',
                 ', '.join(sorted(sequences_with_no_mistakes)))



if __name__ == "__main__":
    main()
