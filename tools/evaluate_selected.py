"""Evaluate mAP on specific sequences."""

import argparse
import collections
import logging
import os
import pickle
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from tabulate import tabulate

import _init_paths  # noqa: F401
from datasets.task_evaluation import (_coco_eval_to_box_results,
                                      _coco_eval_to_mask_results)
from utils.coco import accumulate_with_more_info, get_iou_index
from utils.logging import setup_logging


def evaluate(detection_eval, segmentation_eval):
    with redirect_stdout(open(os.devnull, 'w')):
        detection_eval.evaluate()
        detection_eval.accumulate()
        detection_eval.summarize()

        segmentation_eval.evaluate()
        segmentation_eval.accumulate()
        segmentation_eval.summarize()
    results = _coco_eval_to_box_results(detection_eval)
    segmentation_results = _coco_eval_to_mask_results(segmentation_eval)
    results.update(segmentation_results)
    return results


def simple_table(rows):
    lengths = [
        max(len(row[i]) for row in rows) + 1 for i in range(len(rows[0]))
    ]
    row_format = ' '.join(('{:<%s}' % length) for length in lengths)

    output = ''
    for i, row in enumerate(rows):
        if i > 0:
            output += '\n'
        output += row_format.format(*row)
    return output


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--eval-dir',
        type=Path,
        help=('Evaluation directory containing detection_results.pkl and '
              'segmentation_results.pkl. Alternatively, thsee can be '
              'specified individually with --detections-pickle and '
              '--segmentations-pickle'))
    parser.add_argument(
        '--detections-pickle',
        type=Path,
        help=("Pickle file containing a COCOEval object; usually "
              "'detection_results.pkl'."))
    parser.add_argument(
        '--segmentations-pickle',
        type=Path,
        help=("Pickle file containing a COCOEval object; usually "
              "'segmentation_results.pkl'."))
    parser.add_argument(
        '--evaluate-sequences',
        nargs='*',
        help='Sequences to evaluate on. By default, all sequences.')
    parser.add_argument(
        '--skip-sequences',
        nargs='*',
        help='Sequences to skip. By default, no sequences.')
    parser.add_argument('--output-log', required=True)

    args = parser.parse_args()

    assert args.evaluate_sequences is None or args.skip_sequences is None

    assert args.eval_dir or args.detections_pickle
    assert args.eval_dir or args.segmentations_pickle

    if not args.detections_pickle:
        args.detections_pickle = args.eval_dir / 'detection_results.pkl'
    if not args.segmentations_pickle:
        args.segmentations_pickle = args.eval_dir / 'segmentation_results.pkl'

    output_log = Path(args.output_log)

    setup_logging(str(output_log))
    logging.info('Args:\n%s', vars(args))

    with open(args.detections_pickle, 'rb') as f:
        detection_eval = pickle.load(f)  # COCOeval object
    with open(args.segmentations_pickle, 'rb') as f:
        segmentation_eval = pickle.load(f)  # COCOeval object

    sequence_to_image_ids = collections.defaultdict(list)
    for image_id in detection_eval.cocoGt.imgs.keys():
        sequence = Path(detection_eval.cocoGt.imgs[image_id]['file_name']).parent.name
        sequence_to_image_ids[sequence].append(image_id)
    all_sequences = set(sequence_to_image_ids.keys())

    if args.evaluate_sequences:
        for sequence in args.evaluate_sequences:
            if sequence not in all_sequences:
                raise ValueError('Unknown sequence: %s. Valid sequences: %s' %
                                 (sequence, all_sequences))
        eval_sequences = set(args.evaluate_sequences)
    elif args.skip_sequences:
        for sequence in args.skip_sequences:
            if sequence not in all_sequences:
                raise ValueError('Unknown sequence: %s. Valid sequences: %s' %
                                 (sequence, all_sequences))
        eval_sequences = all_sequences - set(args.skip_sequences)
    else:
        eval_sequences = all_sequences
    eval_sequences = sorted(eval_sequences)

    valid_image_ids = []
    for sequence in eval_sequences:
        valid_image_ids += sequence_to_image_ids[sequence]

    headers = ['Sequence', 'Det mAP', 'Seg mAP', 'Det @ 0.5', 'Seg @ 0.5']
    all_results = []
    for sequence in eval_sequences:
        detection_eval.params.imgIds = sequence_to_image_ids[sequence]
        segmentation_eval.params.imgIds = sequence_to_image_ids[sequence]

        results = evaluate(detection_eval, segmentation_eval)
        all_results.append((sequence, '%.2f' % (100 * results['box']['AP']),
                            '%.2f' % (100 * results['mask']['AP']),
                            '%.2f' % (100 * results['box']['AP50']),
                            '%.2f' % (100 * results['mask']['AP50'])))
    logging.info('Results:\n%s', simple_table([headers] + all_results))

    detection_eval.params.imgIds = valid_image_ids
    segmentation_eval.params.imgIds = valid_image_ids

    results = evaluate(detection_eval, segmentation_eval)
    to_log = [
        ('Det mAP', '%.2f' % (100 * results['box']['AP'])),
        ('Seg mAP', '%.2f' % (100 * results['mask']['AP'])),
        ('Det @ 0.5', '%.2f' % (100 * results['box']['AP50'])),
        ('Seg @ 0.5', '%.2f' % (100 * results['mask']['AP50'])),
    ]
    logging.info(','.join(x[0] for x in to_log))
    logging.info(','.join(str(x[1]) for x in to_log))


if __name__ == "__main__":
    main()
