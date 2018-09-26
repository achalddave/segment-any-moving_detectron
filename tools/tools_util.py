import logging
import numpy as np

import _init_paths
from core.config import cfg
from datasets import dataset_catalog


def update_cfg_for_dataset(datasets, update_pixel_means=False):
    """Update global cfg for working with a list of datasets.

    Note that this list of datasets is used in parallel to load multiple
    images (e.g., flow, RGB, etc.) for the same dataset. Thus, the number of
    classes should be the same for each one.
    """
    num_classes, all_pixel_means, force_test_eval = load_datasets_info(
        datasets)
    cfg.MODEL.NUM_CLASSES = num_classes

    if force_test_eval:
        logging.info(
            "Forcing JSON dataset eval true for datasets: %s" % datasets)
        cfg.TEST.FORCE_JSON_DATASET_EVAL = True

    if update_pixel_means:
        for i, pixel_means in enumerate(all_pixel_means):
            if pixel_means is not None:
                logging.info("Changing pixel mean to zero for dataset '%s'" %
                             datasets[i])
                cfg.PIXEL_MEANS[i] = pixel_means


def load_datasets_info(datasets):
    infos = [load_single_dataset_info(dataset) for dataset in datasets]
    all_num_classes, all_pixel_means, all_force_test_eval = zip(*infos)
    if any(n != all_num_classes[0] for n in all_num_classes[1:]):
        raise ValueError(
            'Inconsistent num classes: %s' % all_num_classes)

    if any(b != all_force_test_eval[0] for b in all_force_test_eval[1:]):
        raise ValueError(
            'Inconsistent FORCE_TEST_EVAL inferred: %s' % all_force_test_eval)

    return all_num_classes[0], all_pixel_means, all_force_test_eval[0]


def load_single_dataset_info(dataset):
    if dataset not in dataset_catalog.DATASETS:
        raise ValueError("Unexpected dataset: %s" % dataset)
    dataset_info = dataset_catalog.DATASETS[dataset]
    if dataset_catalog.NUM_CLASSES not in dataset_info:
        raise ValueError(
            "Num classes not listed in dataset: %s" % dataset)

    pixel_means = None
    if dataset_info[dataset_catalog.IS_FLOW]:
        pixel_means = np.zeros((1, 1, 3))

    force_test_eval = False
    if any(x in dataset for x in ("flyingthings", "fbms", "davis", "ytvos")):
        force_test_eval = True

    num_classes = dataset_info[dataset_catalog.NUM_CLASSES]
    return num_classes, pixel_means, force_test_eval
