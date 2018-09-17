"""Combine two sets of detections using various heuristics."""

import argparse
import logging
import numpy as np
import pickle
from copy import deepcopy
from pathlib import Path

import pycocotools.mask as mask_util

import _init_paths  # noqa: F401
from utils.logging import setup_logging


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


def main():
    with open(__file__, 'r') as f:
        _source = f.read()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--detections1',
        help='detection pickle file',
        required=True)
    parser.add_argument(
        '--detections2',
        help='detection pickle file)',
        required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--method',
        default='merge',
        choices=['merge', 'non-overlapping', 'second-after-first'],
        help="""Method to combine detections. [merge]: Output all detections
        from both detection files. [non-overlapping]: Keep all detections from
        eval1, and detections from eval2 that don't overlap with eval1.
        [second-after-first]: Reduce score of second predictions to be below
        that of first predictions per image.""")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logging_path = str(output_dir / (Path(__file__).stem + '.py.log'))
    setup_logging(logging_path)

    logging.info('Args:\n%s' % args)

    with open(args.detections1, 'rb') as f:
        result1 = Result.from_file(f)

    with open(args.detections2, 'rb') as f:
        result2 = Result.from_file(f)

    if args.method == 'merge':
        merged_result = merge(result1, result2)
    elif args.method == 'non-overlapping':
        merged_result = merge_non_overlapping(result1, result2)
    elif args.method == 'second-after-first':
        merged_result = merge_second_after_first(result1, result2)
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
