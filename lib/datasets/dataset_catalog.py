# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'
NUM_CLASSES = 'num_classes'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'
IS_FLOW = 'is_flow'
# If specified, change the extension of the image file_name.
IMAGE_EXTENSION = 'image_extension'

# Available datasets
DATASETS = {
    'cityscapes_fine_instanceonly_seg_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw',
        IS_FLOW: False
    },
    'cityscapes_fine_instanceonly_seg_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw',
        IS_FLOW: False
    },
    'cityscapes_fine_instanceonly_seg_test': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw',
        IS_FLOW: False
    },
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json',
        NUM_CLASSES: 81,
        IS_FLOW: False
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json',
        NUM_CLASSES: 81,
        IS_FLOW: False
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json',
        NUM_CLASSES: 81,
        IS_FLOW: False
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json',
        NUM_CLASSES: 81,
        IS_FLOW: False
    },
    'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json',
        NUM_CLASSES: 81,
        IS_FLOW: False
    },
    'coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json',
        NUM_CLASSES: 81,
        IS_FLOW: False
    },
    'coco_2017_train_objectness': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017_objectness.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'coco_2017_val_objectness': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017_objectness.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
        NUM_CLASSES: 81,
        IS_FLOW: False
    },
    'coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
        NUM_CLASSES: 81,
        IS_FLOW: False
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        IM_PREFIX:
            'COCO_test2015_',
        NUM_CLASSES: 81,
        IS_FLOW: False
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        IM_PREFIX:
            'COCO_test2015_',
        NUM_CLASSES: 81,
        IS_FLOW: False
    },
    'coco_stuff_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_train.json',
        IS_FLOW: False
    },
    'coco_stuff_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_val.json',
        IS_FLOW: False
    },
    'flyingthings3d_rgb_train': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/rgb_images/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'flyingthings3d_rgb_test': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/rgb_images/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'flyingthings3d_gtflow_train': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/gt_flow_images/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'flyingthings3d_gtflow_test': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/gt_flow_images/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'flyingthings3d_gtflow_3369_subset_train': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/gt_flow_images/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/train-without-last-frame_subset-3369.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'flyingthings3d_estimatedflow_train': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/estimated_flow_images/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'flyingthings3d_estimatedflow_test': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/estimated_flow_images/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'fbms_rgb_train': {
        IM_DIR: _DATA_DIR + '/fbms/highres/rgb/',
        ANN_FN: _DATA_DIR + '/fbms/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'fbms_rgb_test': {
        IM_DIR: _DATA_DIR + '/fbms/highres/rgb/',
        ANN_FN: _DATA_DIR + '/fbms/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'fbms_flow_train': {
        IM_DIR: _DATA_DIR + '/fbms/highres/liteflownet/',
        ANN_FN: _DATA_DIR + '/fbms/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'fbms_flow_test': {
        IM_DIR: _DATA_DIR + '/fbms/highres/liteflownet/',
        ANN_FN: _DATA_DIR + '/fbms/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'fbms_flownet2_train': {
        IM_DIR: _DATA_DIR + '/fbms/highres/flownet2/',
        ANN_FN: _DATA_DIR + '/fbms/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'fbms_flownet2_test': {
        IM_DIR: _DATA_DIR + '/fbms/highres/flownet2/',
        ANN_FN: _DATA_DIR + '/fbms/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'fbms_hed_flow_train': {
        IM_DIR: _DATA_DIR + '/fbms/highres/hed-flow-concat/',
        ANN_FN: _DATA_DIR + '/fbms/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'fbms_hed_flow_test': {
        IM_DIR: _DATA_DIR + '/fbms/highres/hed-flow-concat/',
        ANN_FN: _DATA_DIR + '/fbms/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'fbms_hed_flownet2_train': {
        IM_DIR: _DATA_DIR + '/fbms/highres/hed-flownet2-concat/',
        ANN_FN: _DATA_DIR + '/fbms/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'fbms_hed_flownet2_test': {
        IM_DIR: _DATA_DIR + '/fbms/highres/hed-flownet2-concat/',
        ANN_FN: _DATA_DIR + '/fbms/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_rgb_moving_train': {
        IM_DIR: _DATA_DIR + '/davis/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-train-no-last-frame_jpg-extension.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_rgb_moving_trainval': {
        IM_DIR: _DATA_DIR + '/davis/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-trainval-no-last-frame_jpg-extension.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_rgb_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-test-no-last-frame_jpg-extension.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_hed_moving_train': {
        IM_DIR: _DATA_DIR + '/davis/hed/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_hed_moving_trainval': {
        IM_DIR: _DATA_DIR + '/davis/hed/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-trainval-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_hed_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/hed/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-test-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_flow_relabeled_moving_train': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis/annotations-always-moving/moving-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_flow_relabeled_moving_trainval': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis/annotations-always-moving/moving-trainval-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_flow_relabeled_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis/annotations-always-moving/moving-test-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_flow_moving_train': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_flow_moving_trainval': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-trainval-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_flow_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-test-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_flownet2_flow_moving_train': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_flownet2_flow_moving_trainval': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-trainval-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_flownet2_flow_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-test-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis16_flow_train': {
        IM_DIR: _DATA_DIR + '/davis16/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis16_flow_val': {
        IM_DIR: _DATA_DIR + '/davis16/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis16_rgb_train': {
        IM_DIR: _DATA_DIR + '/davis16/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/train-no-last-frame_jpg-extension.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis16_rgb_val': {
        IM_DIR: _DATA_DIR + '/davis16/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/val-no-last-frame_jpg-extension.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    # The "moving" splits for DAVIS 16 contain all the videos from DAVIS 16,
    # but the splits respect the splits made for DAVIS 17 moving videos above.
    'davis16_flow_moving_train': {
        IM_DIR: _DATA_DIR + '/davis16/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis16/moving-annotations/train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis16_flow_moving_test': {
        IM_DIR: _DATA_DIR + '/davis16/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis16/moving-annotations/test-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis16_rgb_moving_train': {
        IM_DIR: _DATA_DIR + '/davis16/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis16/moving-annotations/train-no-last-frame_jpg-extension.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis16_rgb_moving_test': {
        IM_DIR: _DATA_DIR + '/davis16/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis16/moving-annotations/test-no-last-frame_jpg-extension.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_flow_vis_moving_train': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet-vis/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_flow_vis_moving_trainval': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet-vis/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-trainval-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_flow_vis_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet-vis/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-test-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_flownet2_vis_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/flownet2-vis/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-test-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_hed_flow_moving_train': {
        IM_DIR: _DATA_DIR + '/davis/hed-flow-concat/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_hed_flow_moving_trainval': {
        IM_DIR: _DATA_DIR + '/davis/hed-flow-concat/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-trainval-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_hed_flow_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/hed-flow-concat/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-test-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_hed_flownet2_moving_train': {
        IM_DIR: _DATA_DIR + '/davis/hed-flownet2-concat/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_hed_flownet2_moving_trainval': {
        IM_DIR: _DATA_DIR + '/davis/hed-flownet2-concat/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-trainval-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_hed_flownet2_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/hed-flownet2-concat/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-test-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'ytvos_rgb_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/rgb/train',
        ANN_FN: _DATA_DIR + '/ytvos/train-splits/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
    },
    'ytvos_rgb_all_moving_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/rgb/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
    },
    'ytvos_rgb_all_moving_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/rgb/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
    },
    'ytvos_rgb_all_moving_strict_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/rgb/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
    },
    'ytvos_rgb_all_moving_strict_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/rgb/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
    },
    'ytvos_rgb_all_moving_strict_interesting_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/rgb/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/interesting-only/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
    },
    'ytvos_flow_vis_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet-vis/train',
        ANN_FN: _DATA_DIR + '/ytvos/train-splits/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_vis_all_moving_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet-vis/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_vis_all_moving_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet-vis/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_vis_all_moving_strict_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet-vis/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_vis_all_moving_strict_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet-vis/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_vis_all_moving_strict_interesting_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet-vis/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/interesting-only/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet/train',
        ANN_FN: _DATA_DIR + '/ytvos/train-splits/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_all_moving_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_all_moving_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_all_moving_strict_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_all_moving_strict_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flow_all_moving_strict_interesting_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/liteflownet/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/interesting-only/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_hed_flow_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/hed-flow-concat/train',
        ANN_FN: _DATA_DIR + '/ytvos/train-splits/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_hed_flow_all_moving_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/hed-flow-concat/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_hed_flow_all_moving_strict_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/hed-flow-concat/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_hed_flow_all_moving_strict_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/hed-flow-concat/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_hed_flow_all_moving_strict_interesting_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/hed-flow-concat/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/interesting-only/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flownet2_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/flownet2/train',
        ANN_FN: _DATA_DIR + '/ytvos/train-splits/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flownet2_all_moving_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/flownet2/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flownet2_all_moving_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/flownet2/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flownet2_all_moving_strict_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/flownet2/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flownet2_all_moving_strict_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/flownet2/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_flownet2_all_moving_strict_interesting_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/flownet2/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/interesting-only/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True,
        IMAGE_EXTENSION: '.png'
    },
    'keypoints_coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'keypoints_coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'keypoints_coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'keypoints_coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'keypoints_coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'keypoints_coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'keypoints_coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'keypoints_coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'keypoints_coco_2017_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'voc_2007_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007',
        IS_FLOW: False
    },
    'voc_2007_test': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007',
        IS_FLOW: False
    },
    'voc_2012_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012',
        IS_FLOW: False
    }
}

for dataset, info in DATASETS.items():
    if IMAGE_EXTENSION not in info:
        info[IMAGE_EXTENSION] = None
