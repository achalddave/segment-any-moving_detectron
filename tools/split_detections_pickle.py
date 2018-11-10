"""Split detections.pkl into into multiple pickle files, one per image.

This converts the output of test_net.py into a format similar to
infer_simple.py, allowing scripts to operate on both in the same way.
"""

import argparse
import logging
import pickle
from pathlib import Path

from tqdm import tqdm

import _init_paths
from utils.logging import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--detections-pickle',
        type=Path,
        required=True,
        help='detections.pkl file, as output by tools/test_net.py')
    parser.add_argument(
        '--output-dir',
        default='{pickle_parent}/split-pickles/',
        help='Output directory')
    parser.add_argument(
        '--coco-eval-pickle',
        type=Path,
        help=('Pickle file containing CocoEval object; by default, it is '
              'inferred as {pickle_parent}/detection_results.pkl. This is '
              'used to get the image names in the same order as in '
              'detections.pkl. Alternatively, the --annotations-json used '
              'with test_net.py can be specified. Mutually exclusive with '
              '--annotations-json.'))
    parser.add_argument(
        '--annotations-json',
        type=Path,
        help=('COCO annotations JSON file. By default, this is inferred from '
              'the --coco-eval-pickle file, which itself is inferred from '
              '--detections-pickle. Mutually exclusive with '
              '--coco-eval-pickle.'))

    args = parser.parse_args()
    args.output_dir = Path(
        args.output_dir.format(pickle_parent=args.detections_pickle.parent))
    args.output_dir.mkdir(exist_ok=True, parents=True)

    setup_logging(str(args.output_dir / (Path(__file__).name + '.log')))

    logging.info('Args:\n%s', vars(args))

    assert ((args.coco_eval_pickle is None)
            or (args.annotations_json is None)), (
                'At most one of --coco-eval-pickle or --annotations-json can '
                'be specified.')
    if args.annotations_json is not None:
        from pycocotools.coco import COCO
        groundtruth = COCO(str(args.annotations_json))
    else:
        if args.coco_eval_pickle is None:
            args.coco_eval_pickle = '{pickle_parent}/detection_results.pkl'
        args.coco_eval_pickle = args.coco_eval_pickle.format(pickle_parent=args.detections_pickle.parent)
        with open(args.coco_eval_pickle, 'rb') as f:
            groundtruth = pickle.load(f).cocoGt

    image_ids = sorted(groundtruth.getImgIds())

    with open(args.detections_pickle, 'rb') as f:
        data = pickle.load(f)
        boxes = data['all_boxes']
        masks = data['all_segms']
        keypoints = data['all_keyps']
        num_classes = len(boxes)
        for c in range(num_classes):
            assert len(boxes[c]) == len(image_ids), (
                f'Expected {len(image_ids)} boxes for class {c}, got '
                f'{len(boxes[c])}')
        for i, image_id in enumerate(tqdm(image_ids)):
            output = {
                'boxes': [boxes[c][i] for c in range(num_classes)],
                'segmentations': [masks[c][i] for c in range(num_classes)],
                'keypoints': [keypoints[c][i] for c in range(num_classes)]
            }
            output_file = (
                args.output_dir / groundtruth.imgs[image_ids[i]]['file_name']
            ).with_suffix('.pickle')
            output_file.parent.mkdir(exist_ok=True, parents=True)
            with open(output_file, 'wb') as f:
                pickle.dump(output, f)

if __name__ == "__main__":
    main()
