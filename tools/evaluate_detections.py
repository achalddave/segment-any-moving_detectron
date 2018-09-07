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
    parser.add_argument('--detections', required=True)
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

    if args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
    elif args.dataset == 'coco_2017_objectness':
        cfg.TEST.DATASETS = ('coco_2017_val_objectness',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'flyingthings':
        cfg.TEST.DATASETS = ('flyingthings3d_test',)
    elif args.dataset == 'flyingthings_train':
        cfg.TEST.DATASETS = ('flyingthings3d_train',)
    elif args.dataset == 'flyingthings_estimatedflow':
        cfg.TEST.DATASETS = ('flyingthings3d_estimatedflow_test',)
    elif args.dataset == 'flyingthings_estimatedflow_train':
        cfg.TEST.DATASETS = ('flyingthings3d_estimatedflow_train',)
    elif args.dataset == "fbms_flow":
        cfg.TEST.DATASETS = ("fbms_flow_test",)
    elif args.dataset == "fbms_flow_train":
        cfg.TEST.DATASETS = ("fbms_flow_train",)
    elif args.dataset == "davis_flow_moving":
        cfg.TEST.DATASETS = ("davis_flow_moving_test",)
    elif args.dataset is not None:
        raise ValueError('Unknown --dataset: %s' % args.dataset)
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'

    if any(x in args.dataset for x in ('flyingthings', 'fbms', 'davis')):
        cfg.TEST.FORCE_JSON_DATASET_EVAL = True

    assert_and_infer_cfg()

    dataset = cfg.TEST.DATASETS[0]

    dataset = load_dataset(dataset)
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


if __name__ == "__main__":
    main()
