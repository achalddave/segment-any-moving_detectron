import logging
import numpy as np

import _init_paths
from core.config import cfg
from datasets import dataset_catalog


def update_cfg_for_dataset(dataset, update_pixel_means=False):
    num_classes, pixel_means, force_test_eval = load_dataset_info(dataset)

    cfg.MODEL.NUM_CLASSES = num_classes
    if force_test_eval:
        logging.info(
            "Forcing JSON dataset eval true for dataset '%s'" % dataset)
        cfg.TEST.FORCE_JSON_DATASET_EVAL = True

    if update_pixel_means and pixel_means is not None:
        logging.info("Changing pixel mean to zero for dataset '%s'" % dataset)
        cfg.PIXEL_MEANS = pixel_means


def load_dataset_info(dataset):
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
