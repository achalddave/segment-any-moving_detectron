import argparse
import logging
import pickle
from pathlib import Path
from pprint import pformat

import _init_paths  # pylint: disable=unused-import
import utils.logging
from core.config import cfg, merge_cfg_from_file, assert_and_infer_cfg
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
        help='Default: parent_dir({{detections}})/{{dataset}}-eval-out"')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(
            args.detections).parent / ('%s-eval-out' % args.dataset)
        args.output_dir.mkdir(exist_ok=True)
    logging_path = str(args.output_dir / 'evaluate-detections.log')
    utils.logging.setup_logging(logging_path)
    logging.info('Args: %s', pformat(vars(args)))

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)

    if args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'flyingthings':
        cfg.TEST.DATASETS = ('flyingthings3d_test',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'flyingthings_train':
        cfg.TEST.DATASETS = ('flyingthings3d_train',)
        cfg.MODEL.NUM_CLASSES = 2
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    dataset = cfg.TEST.DATASETS[0]

    dataset = load_dataset(dataset)
    with open(args.detections, 'rb') as f:
        data = pickle.load(f)
    results = task_evaluation.evaluate_all(dataset, data['all_boxes'],
                                           data['all_segms'],
                                           data['all_keyps'], args.output_dir)
    logging.info('Results:')
    logging.info(results)
    task_evaluation.log_copy_paste_friendly_results(results)


if __name__ == "__main__":
    main()
