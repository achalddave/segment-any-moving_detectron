#!/usr/bin/env python

import argparse
import re
from pathlib import Path


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


# @profile
def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('dataset_glob')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    model_dir = args.model_dir
    steps_with_paths = [(int(x.stem.split('step')[1]), x)
                        for x in model_dir.glob('step*')]
    steps_with_paths = sorted(steps_with_paths, key=lambda x: x[0])

    print('Evaluating model: %s' % model_dir)

    header = None
    regex = re.compile('.*copypaste1: ([^,]*),([^,]*),([^,]*),([^,]*).*')

    results = []
    evaluation_files = []
    for step, step_dir in steps_with_paths:
        dataset_dir = list(step_dir.glob(args.dataset_glob))
        if len(dataset_dir) > 1:
            raise ValueError('Multiple directories match dataset_glob:\n%s'
                             % '\n'.join(map(str, dataset_dir)))
        elif len(dataset_dir) == 0:
            continue
        else:
            dataset_dir = dataset_dir[0]

        evaluation_file = dataset_dir / 'evaluation.log'
        if not evaluation_file.exists():
            continue
        evaluation_files.append(evaluation_file)

        with open(evaluation_file, 'r') as f:
            lines = f.readlines()
        current_results = []
        for line in lines:
            match = regex.match(line)
            if match:
                current_results.append(match.groups())
        if len(current_results) != 2:
            if args.verbose:
                print("Couldn't parse evaluation: %s" % evaluation_file)
            continue
        if header is None:
            header = ['Step'] + list(current_results[0])
        results.append([str(step)] + list(current_results[1]))

    if args.verbose:
        print('Evaluation paths:\n%s' % '\n'.join(str(x) for x in evaluation_files))
    print(simple_table([header] + results))


if __name__ == "__main__":
    main()
