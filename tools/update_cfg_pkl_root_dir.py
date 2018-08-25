"""Update root directory in cfg_and_args.pkl from training runs.

This is necessary when the root directory of detectron moves after a model has
been trained. You might need this if you get "No such file or directory"
errors when running `train_net.py`, for example. For the simplest usage,
just run

python tools/update_cfg_pkl_root_dir \
    --input-cfg-and-args path/to/cfg_and_args.pkl
"""

import argparse
import pickle
import logging
import shutil
import yaml
from pathlib import Path

import _init_paths  # noqa: F401
# Necessary to load configs
import utils.collections  # noqa: F401
# End imports necessary to load configs

from utils.logging import setup_logging

DEFAULT_NEW_ROOT = Path(__file__).parent.parent.resolve()


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-cfg-and-args', required=True)
    parser.add_argument(
        '--backup-file',
        default=None,
        help=('Where to backup input pickle. Default: Add ".bak" to input '
              'file name.'))
    parser.add_argument(
        '--new-root',
        default=str(DEFAULT_NEW_ROOT),
        help='New root directory.')

    args = parser.parse_args()

    # We load this first to avoid logging if the input file doesn't exist.
    with open(args.input_cfg_and_args, 'rb') as f:
        data = pickle.load(f)

    cfg = yaml.load(data['cfg'])

    if Path(cfg['ROOT_DIR']) == Path(args.new_root):
        print('Current ROOT_DIR already points to new ROOT_DIR. Exiting '
              'without doing anything. Current: (%s), new: (%s)' %
              (cfg['ROOT_DIR'], args.new_root))

    setup_logging(args.input_cfg_and_args + '.log')

    if args.backup_file is None:
        args.backup_file = args.input_cfg_and_args + '.bak'

    logging.info('Args:\n%s' % vars(args))

    shutil.move(args.input_cfg_and_args, args.backup_file)

    if Path(cfg['ROOT_DIR']) != args.new_root:
        cfg['ROOT_DIR'] = args.new_root
        logging.info(
            'Updating ROOT_DIR in loaded config to '
            'current ROOT_DIR: %s' % cfg['ROOT_DIR'])

    data['cfg'] = yaml.dump(cfg)
    with open(args.input_cfg_and_args, 'wb') as f:
        data = pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
