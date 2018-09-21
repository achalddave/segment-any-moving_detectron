import argparse
import logging
import pickle
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

import _init_paths  # noqa: F401
from utils.coco import accumulate_with_more_info, get_iou_index
from utils.logging import setup_logging


def log_detection_eval_metrics(coco_eval):
    # Modified from json_dataset_evaluator._log_detection_eval_metrics
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95
    ind_lo = get_iou_index(coco_eval, IoU_lo_thresh)
    ind_hi = get_iou_index(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    logging.info(
        '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
            IoU_lo_thresh, IoU_hi_thresh))
    logging.info('{:.1f}'.format(100 * ap_default))
    for index in range(len(coco_eval.cocoGt.cats)):
        # minus 1 because of __background__
        precision = coco_eval.eval['precision'][
            ind_lo:(ind_hi + 1), :, index - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        logging.info('{:.1f}'.format(100 * ap))
    logging.info('~~~~ Summary metrics ~~~~')
    summary_f = StringIO()
    with redirect_stdout(summary_f):
        coco_eval.summarize()
    logging.info('\n%s' % summary_f.getvalue())


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--coco-eval-pickle', required=True)
    parser.add_argument('--output-log', required=True)

    args = parser.parse_args()

    coco_eval = Path(args.coco_eval_pickle)
    assert coco_eval.exists()
    output_log = Path(args.output_log)

    setup_logging(str(output_log))
    logging.info('Args:\n%s', vars(args))

    with open(args.coco_eval_pickle, 'rb') as f:
        coco_eval = pickle.load(f)
    # log_detection_eval_metrics(coco_eval)

    # has shape (num_iou_thresh, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    outputs = accumulate_with_more_info(coco_eval)
    precision = outputs['precision'][:, :, :, 0, 2]
    # has shape (num_iou_thresh, cls, area range, max dets)
    recall = outputs['recall'][:, :, :, 0, 2]
    scores = outputs['scores'][:, :, :, 0, 2]

    iou_thresh = 0.5
    iou_index = get_iou_index(coco_eval, iou_thresh)

    # Scores are sorted in descending order
    scores = scores[iou_index].squeeze()
    score_thresh = 0.95
    score_index = np.where(scores < score_thresh)[0][0] - 1

    logging.info(f'Precision at IOU={iou_thresh}, Score={score_thresh}')
    logging.info(precision[iou_index, score_index])

    logging.info(f'Recall at IOU={iou_thresh}, Score={score_thresh}')
    logging.info(recall[iou_index, score_index])


if __name__ == "__main__":
    main()
