"""Combine two sets of detections using various heuristics."""

import argparse
import logging
import numpy as np
import pickle
import pprint
from copy import deepcopy
from pathlib import Path

import pycocotools.mask as mask_util

import _init_paths  # noqa: F401
from utils.logging import setup_logging


def translate_range(value, old_range, new_range):
    """
    Translate value in one range to another. Useful for scaling scores.

    >>> translate_range(0.5, (0, 1), (0, 2))
    1.0
    >>> translate_range(1, (0, 1), (0, 2))
    2.0
    >>> translate_range(3, (2, 4), (5, 6))
    5.5
    >>> translate_range(0.5, (0, 1), (0, 2))
    1.0
    >>> translate_range(np.array([2, 2.5, 3, 3.5, 4]), (2, 4), (5, 7)).tolist()
    [5.0, 5.5, 6.0, 6.5, 7.0]
    >>> translate_range(np.array([2, 2.5, 3, 3.5, 4]), (2, 4), (5, 5)).tolist()
    [5.0, 5.0, 5.0, 5.0, 5.0]
    """
    value = np.asarray(value)
    old_min, old_max = old_range

    if np.any(value < old_min):
        raise ValueError('Value(s) (%s) < min(old_range) (%s)' %
                         (value[value < old_min], old_min))

    if np.any(value > old_max):
        raise ValueError('Value(s) (%s) > max(old_range) (%s)' %
                         (value[value > old_max], old_max))

    if (old_max - old_min) < 1e-10:
        return old_max

    new_min, new_max = new_range
    scale = (new_max - new_min) / (old_max - old_min)
    return (value - old_min) * scale + new_min


class Result:
    def __init__(self, boxes, masks, keypoints):
        """
        Args:
            boxes (list): boxes[c][i] contains a numpy array of boxes for class
                c, image i.
            masks (list): masks[c][i] contains a list of segmentations for
                class c, image i.
            keypoints (list): keypoints[c][i] contains keypoints for class c,
                image i. I'm not sure of the format as I've never used it.
        """
        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints

        self.num_classes = len(self.boxes)
        self.num_images = len(self.boxes[0])

        assert len(self.masks) == self.num_classes
        assert len(self.masks[0]) == self.num_images
        assert len(self.keypoints) == self.num_classes
        assert len(self.keypoints[0]) == self.num_images

    @staticmethod
    def from_file(f):
        d = pickle.load(f)
        return Result(d['all_boxes'], d['all_segms'], d['all_keyps'])

    def to_file(self, f):
        return pickle.dump({
            'all_boxes': self.boxes,
            'all_segms': self.masks,
            'all_keyps': self.keypoints
        }, f)

    @staticmethod
    def empty_result(num_classes, num_images):
        """Create an empty Result object with the right structure."""
        return Result(
            Result._empty_result_list(num_classes, num_images),
            Result._empty_result_list(num_classes, num_images),
            Result._empty_result_list(num_classes, num_images))

    @staticmethod
    def _empty_result_list(num_classes, num_images):
        return [[None for _ in range(num_images)] for _ in range(num_classes)]


def merge(first, second):
    """
    Args:
        first (Result)
        second (Result)

    Returns:
        merged (Result)
    """
    assert first.num_classes == second.num_classes
    assert first.num_images == second.num_images
    new_result = Result.empty_result(first.num_classes, first.num_images)
    for c in range(first.num_classes):
        for i in range(first.num_images):
            new_result.boxes[c][i] = np.vstack((first.boxes[c][i],
                                                second.boxes[c][i]))
            new_result.masks[c][i] = first.masks[c][i] + second.masks[c][i]
            new_result.keypoints[c][i] = (
                first.keypoints[c][i] + second.keypoints[c][i])
    return new_result


def merge_non_overlapping(first, second):
    """
    Merge results, discarding any overlapping predictions from second.

    Args:
        first (Result)
        second (Result)

    Returns:
        merged (Result)
    """
    assert first.num_classes == second.num_classes
    assert first.num_images == second.num_images
    num_classes = first.num_classes
    num_images = first.num_images

    output_results = Result.empty_result(num_classes, num_images)
    for c in range(num_classes):
        for i in range(num_images):
            masks1 = first.masks[c][i]
            masks2 = second.masks[c][i]

            if not masks2:
                output_results.boxes[c][i] = first.boxes[c][i]
                output_results.masks[c][i] = first.masks[c][i]
                output_results.keypoints[c][i] = first.keypoints[c][i]
                continue

            if not masks1:
                output_results.boxes[c][i] = second.boxes[c][i]
                output_results.masks[c][i] = second.masks[c][i]
                output_results.keypoints[c][i] = second.keypoints[c][i]
                continue

            ious = mask_util.iou(
                masks1, masks2, pyiscrowd=np.zeros(len(second.masks)))
            # List of indices of predictions to keep from second result
            valid_indices = [
                m for m in range(len(masks2)) if np.all(ious[:, m] < 0.8)
            ]
            valid_boxes = second.boxes[c][i][valid_indices]
            valid_masks = [second.masks[c][i][m] for m in valid_indices]
            if second.keypoints[c][i]:
                valid_keypoints = [
                    second.keypoints[c][i][m] for m in valid_indices
                ]
            else:
                valid_keypoints = []

            output_results.boxes[c][i] = np.vstack((first.boxes[c][i],
                                                    valid_boxes))
            output_results.masks[c][i] = first.masks[c][i] + valid_masks
            output_results.keypoints[c][i] = (
                first.keypoints[c][i] + valid_keypoints)
            assert (output_results.boxes[c][i].shape[0] == len(
                output_results.masks[c][i]))
    return output_results


def merge_second_after_first(first, second):
    """
    Merge results, putting second predictions below first for each image.

    Args:
        first (Result)
        second (Result)

    Returns:
        merged (Result)
    """
    assert first.num_classes == second.num_classes
    assert first.num_images == second.num_images
    num_classes = first.num_classes
    num_images = first.num_images

    second = deepcopy(second)
    for c in range(num_classes):
        for i in range(num_images):
            if len(first.boxes[c][i]):
                min_score1 = first.boxes[c][i][:, 4].min()
                second.boxes[c][i][:, 4] *= min_score1
    return merge(first, second)


def merge_overlap1_nonoverlap1(first, second):
    """
    Output only detections from first, moving non-overlapping detections last.

    The ranking is ordered as follows:
        - detections from first that overlap with second
        - detections from first that don't overlap with second
    """
    assert first.num_classes == second.num_classes
    assert first.num_images == second.num_images
    num_classes = first.num_classes
    num_images = first.num_images

    output_results = Result.empty_result(num_classes, num_images)
    min_score = float('inf')
    max_score = float('-inf')

    for c in range(num_classes):
        for i in range(num_images):
            if len(first.boxes[c][i]) > 0:
                first_scores = first.boxes[c][i][:, 4]
                min_score = min(min_score, first_scores.min())
                max_score = max(max_score, first_scores.max())

    assert np.isfinite(min_score)
    assert np.isfinite(max_score)

    # Split range of [min_score, max_score] into two ranges:
    # [min_score, mid_score), [mid_score, max_score]. The former range will
    # contain non-overlapping detections, while the latter contains overlapping
    # detections.
    mid_score = (max_score + min_score) / 2

    IOU_THRESH = 0.8
    for c in range(num_classes):
        for i in range(num_images):
            masks1 = first.masks[c][i]
            masks2 = second.masks[c][i]

            if not masks2 or not masks1:
                output_results.boxes[c][i] = first.boxes[c][i]
                output_results.masks[c][i] = first.masks[c][i]
                output_results.keypoints[c][i] = first.keypoints[c][i]
                continue

            # (num_masks1, num_masks2)
            ious = mask_util.iou(
                masks1, masks2, pyiscrowd=np.zeros(len(second.masks)))

            # (num_masks1,) array
            overlapping_first_binary = np.any(ious > IOU_THRESH, axis=1)
            overlapping_first = np.where(overlapping_first_binary)[0]
            nonoverlapping_first = np.where(~overlapping_first_binary)[0]

            overlapping_boxes = first.boxes[c][i][overlapping_first]
            nonoverlapping_boxes = first.boxes[c][i][nonoverlapping_first]

            overlapping_boxes[:, 4] = translate_range(
                overlapping_boxes[:, 4], (min_score, max_score),
                (mid_score, max_score))
            nonoverlapping_boxes[:, 4] = translate_range(
                nonoverlapping_boxes[:, 4], (min_score, max_score),
                (min_score, mid_score))

            overlapping_masks = [
                first.masks[c][i][m] for m in overlapping_first
            ]
            nonoverlapping_masks = [
                first.masks[c][i][m] for m in nonoverlapping_first
            ]

            if first.keypoints[c][i]:
                overlapping_keypoints = [
                    first.keypoints[c][i][m] for m in overlapping_first
                ]
                nonoverlapping_keypoints = [
                    first.keypoints[c][i][m] for m in nonoverlapping_first
                ]
            else:
                overlapping_keypoints = []
                nonoverlapping_keypoints = []

            if (len(overlapping_masks) + len(nonoverlapping_masks) !=
                    first.boxes[c][i].shape[0]):
                __import__('ipdb').set_trace()
            output_results.boxes[c][i] = np.vstack((overlapping_boxes,
                                                    nonoverlapping_boxes))
            output_results.masks[c][i] = (
                overlapping_masks + nonoverlapping_masks)
            output_results.keypoints[c][i] = (
                overlapping_keypoints + nonoverlapping_keypoints)
            assert (output_results.boxes[c][i].shape[0] == len(
                output_results.masks[c][i]))
    return output_results


def main():
    with open(__file__, 'r') as f:
        _source = f.read()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--detections1',
        help='detection pickle file',
        required=True)
    parser.add_argument(
        '--detections2',
        help='detection pickle file',
        required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--method',
        default='merge',
        choices=[
            'all1+all2', 'all1+nonoverlap2', 'overlap1>nonoverlap1',
            'overlap1>nonoverlap1>nonoverlap2', 'all1>all2'
        ],
        help=('Method to combine detections.\n'
              """[Selection]: "all" indicates all detections. "overlap1"
              indicates detections from detections1 that overlap with a
              detection from detections2.  "nonoverlap1" = all1 - overlap1."""
              '\n'
              """[Combination]: "+" indicates that the two detections are
              concatenated into a single list, keeping the original scores.
              "x>y" indicates that x and y are concatenated, with x set to rank
              above y."""
              '\n'
              """Putting this all together, "overlap1>nonoverlap1>nonoverlap2"
              ranks detections from detections1 that overlap with detections2
              highest; then, detections from detections1 that don't overlap
              with detections2; then, detections detections2 that don't overlap
              with detections1."""))

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logging_path = str(output_dir / (Path(__file__).stem + '.py.log'))
    setup_logging(logging_path)

    logging.info('Args:\n%s' % pprint.pformat(vars(args)))

    with open(args.detections1, 'rb') as f:
        result1 = Result.from_file(f)

    with open(args.detections2, 'rb') as f:
        result2 = Result.from_file(f)

    if args.method == 'all1+all2':
        merged_result = merge(result1, result2)
    elif args.method == 'all1+nonoverlap2':
        merged_result = merge_non_overlapping(result1, result2)
    elif args.method == 'all1>all2':
        merged_result = merge_second_after_first(result1, result2)
    elif args.method == 'overlap1>nonoverlap1':
        merged_result = merge_overlap1_nonoverlap1(result1, result2)
    else:
        raise NotImplementedError

    with open(output_dir / 'detections.pkl', 'wb') as f:
        merged_result.to_file(f)
    file_logger = logging.getLogger(logging_path)

    file_logger.info('Source:')
    file_logger.info('=======')
    file_logger.info(_source)


if __name__ == "__main__":
    main()
