import argparse
import logging
import pickle
from pathlib import Path

import _init_paths  # noqa: F401

import utils.logging
from utils.coco import accumulate_with_more_info, get_iou_index
from utils.env import set_up_matplotlib


def overwrite_or_error(path, overwrite):
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise ValueError('%s already exists.' % path)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--eval-pickle',
        required=True,
        nargs='+',
        help=('Pickle file(s) containing CocoEval object. E.g. '
              '"detection_results.pkl" or "segmentation_results.pkl. If there '
              'are multiple files, each will be plotted as a separate line.'))
    parser.add_argument(
        '--names', nargs='*', help='Names to use for each line in legend.')
    parser.add_argument(
        '--category',
        default=1,
        help='Category to plot pr for. Should be >0 (0 is background).',
        type=int)
    parser.add_argument(
        '--iou-thresh',
        choices=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        default=0.5,
        type=float)
    parser.add_argument(
        '--output-png',
        default='{eval_pickle_parent}/pr_curve_{category}_iou_{iou}.png',
        help=('Output png file. Can contain the following variables: '
              '{eval_pickle_parent}, which will be replaced by the parent dir '
              'of --eval-pickle (the first one if there are mutli); '
              '{category}, which will be replaced by --category; and {iou}, '
              'which will be replaced by --iou-thresh'))
    parser.add_argument(
        '--title',
        default='PR Plot, Category {category}, IoU {iou}',
        help=('Title for plot. Can contain {category} and {iou} specifiers, '
              'as in --output-png'))
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    if args.names:
        assert len(args.names) == len(args.eval_pickle)
    elif len(args.eval_pickle) > 1:
        args.names = [str(x) for x in range(len(args.eval_pickle))]

    set_up_matplotlib()
    assert args.category > 0, (
        '--category should be > 0; 0th category is background')

    output_png = Path(
        args.output_png.format(
            eval_pickle_parent=Path(args.eval_pickle[0]).parent,
            category=args.category,
            iou=args.iou_thresh))

    logging_path = output_png.with_suffix('.log')
    overwrite_or_error(output_png, args.overwrite)
    overwrite_or_error(logging_path, args.overwrite)

    utils.logging.setup_logging(str(logging_path))

    all_precisions = []
    all_recalls = []
    for coco_eval_path in args.eval_pickle:
        with open(coco_eval_path, 'rb') as f:
            coco_eval = pickle.load(f)

        outputs = accumulate_with_more_info(coco_eval)

        iou_index = get_iou_index(coco_eval, args.iou_thresh)

        category = args.category
        # has shape (num_iou_thresh, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = outputs['precision'][iou_index, :, category - 1, 0, 2]
        # has shape (num_iou_thresh, recall, cls, area range, max dets)
        recall = outputs['recall'][iou_index, :, category - 1, 0, 2]

        # Recalls should be in ascending order
        for i in range(1, recall.shape[0]):
            if recall[i] < recall[i-1]:
                recall[i] = recall[i-1]

        all_recalls.append(recall)
        all_precisions.append(precision)

    from matplotlib import pyplot as plt
    for i, (recall, precision) in enumerate(zip(all_recalls, all_precisions)):
        plt.plot(recall, precision, 'o-', markersize=3, label=args.names[i])

    if len(args.eval_pickle) > 1:
        plt.legend()

    plt.title(args.title.format(category=category, iou=args.iou_thresh))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(output_png)
    logging.info('Saved pr plot to %s.' % output_png)


if __name__ == "__main__":
    main()
