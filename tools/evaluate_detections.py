import argparse
import logging
import pickle
from pathlib import Path
from pprint import pformat

import _init_paths  # pylint: disable=unused-import

import utils.logging
from core.config import cfg, merge_cfg_from_file, assert_and_infer_cfg
from core.test_engine import collapse_categories
from datasets import load_dataset, task_evaluation


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument(
        '--detections', required=True, help='detections.pkl file')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True, help='optional config file')
    parser.add_argument(
        '--output-dir',
        default='{detections_parent}/eval-{dataset}',
        help=('Output directory. Can contain the following variables: '
              '{detections_parent}, which will be replaced by the parent dir '
              'of --detections, and {dataset], which will be replaced by '
              '--dataset'))
    parser.add_argument(
        '--collapse-categories',
        action='store_true',
        help=('Collapse all predicted categories to one category. Useful for '
              'evaluating object-specific detectors on objectness datasets.'))

    args = parser.parse_args()

    output_dir = Path(args.output_dir.format(
        detections_parent=Path(args.detections).parent, dataset=args.dataset))
    output_dir.mkdir(exist_ok=True)
    logging_path = str(output_dir / 'evaluate-detections.log')
    utils.logging.setup_logging(logging_path)
    logging.info('Args: %s', pformat(vars(args)))

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)

    dataset = load_dataset(args.dataset)
    cfg.MODEL.NUM_CLASSES = dataset.num_classes
    cfg.TEST.DATASETS = (args.dataset, )
    if any(x in args.dataset
            for x in ("flyingthings", "fbms", "davis", "ytvos")):
        logging.info("Forcing JSON dataset eval true for dataset '%s'" %
                     args.dataset)
        cfg.TEST.FORCE_JSON_DATASET_EVAL = True

    assert_and_infer_cfg()

    with open(args.detections, 'rb') as f:
        data = pickle.load(f)

    if args.collapse_categories:
        def flatten(lst):
            return [x for y in lst for x in y]

        if len(data['all_boxes']) > 2:
            data['all_boxes'], data['all_segms'], data['all_keyps'] = (
                collapse_categories(data['all_boxes'], data['all_segms'],
                                    data['all_keyps']))
    results = task_evaluation.evaluate_all(dataset, data['all_boxes'],
                                           data['all_segms'],
                                           data['all_keyps'], output_dir)
    logging.info('Results:')
    logging.info(results)
    task_evaluation.log_copy_paste_friendly_results(results)

    results = results[args.dataset]
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

    logging.info(','.join(x[0] for x in to_log))
    logging.info(','.join(str(x[1]) for x in to_log))


if __name__ == "__main__":
    main()
