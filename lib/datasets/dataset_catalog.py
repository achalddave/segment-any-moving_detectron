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

DATASETS.update({
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
})

# FBMS
# Paths:
#   /fbms/annotations/ ->
#       /data/achald/track/FBMS/json-annotations/png-filenames/
DATASETS.update({
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
})

# FlyingThings3D
# Paths:
#   /flyingthings3d/annotations ->
#       /data/achald/track/FlyingThings3D/json-annotations/annotations/
DATASETS.update({
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
            _DATA_DIR + '/flyingthings3d/gt_flow/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'flyingthings3d_gtflow_test': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/gt_flow/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'flyingthings3d_gtflow_3369_subset_train': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/gt_flow/',
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
    'flyingthings3d_flownet2_train': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/flownet2/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'flyingthings3d_flownet2_test': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/flownet2/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'flyingthings3d_flownet2_vis_train': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/flownet2-vis/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'flyingthings3d_flownet2_vis_test': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/flownet2-vis/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    # Flownet2 trained on FlyingChairs and FlyingThings3D only.
    'flyingthings3d_flownet2_chairsthings_train': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/flownet2_chairsthings/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/train-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'flyingthings3d_flownet2_chairsthings_test': {
        IM_DIR:
            _DATA_DIR + '/flyingthings3d/flownet2_chairsthings/',
        ANN_FN:
            _DATA_DIR + '/flyingthings3d/annotations/test-without-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    }
})

# DAVIS 17:
#   davis_{split}_{input}_moving:
#     - Sequences where all moving objects are labeled and all static objects
#       are unlabeled, where {split} corresponds to the official DAVIS '17
#       splits. Our final DAVIS '17 number, then, is computed using
#       davis_val_rgb_moving and davis_val_flownet2_flow_moving.
#
# Path mapping:
#   /davis/annotations ->
#       /data/achald/track/DAVIS/2017/moving-only/splits/json-annotation
DATASETS.update({
    'davis_rgb_moving_train': {
        IM_DIR: _DATA_DIR + '/davis/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IMAGE_EXTENSION: '.jpg',
        IS_FLOW: False
    },
    'davis_rgb_moving_trainval': {
        IM_DIR: _DATA_DIR + '/davis/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-trainval-no-last-frame.json',
        NUM_CLASSES: 2,
        IMAGE_EXTENSION: '.jpg',
        IS_FLOW: False
    },
    # Moving sequence from DAVIS 2017's validation set. This is _different_
    # from davis_rgb_moving_val, which is a separate "validation" set I made
    # from the subset of sequences I marked as moving.
    'davis_val_rgb_moving': {
        IM_DIR: _DATA_DIR + '/davis/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis/annotations/davis-val-moving-all-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.jpg',
    },
    'davis_val_flownet2_flow_moving': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis/annotations/davis-val-moving-all-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
})

# DAVIS 16:
#   davis16_{input}_{split}
#     - Contain exactly all the sequences from the official {split} in DAVIS
#       '16. Annotations come from DAVIS 2016, and so contain only a binary
#       foreground-background segmentation mask.
#   davis16_{input}_{split}_instance
#     - Like davis16_{input}_{split}, but annotations come from DAVIS 2017.
#       Since DAVIS '17 contains all the sequences from DAVIS '16, we can take
#       the instance annotations from DAVIS '16 and apply them directly to
#       DAVIS '17.
#
# Path mapping:
#   /davis16/annotations ->
#       /data/achald/track/DAVIS/2016/json-annotations
DATASETS.update({
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
    'davis16_flownet2_train': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis16_flownet2_val': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis16_rgb_train': {
        IM_DIR: _DATA_DIR + '/davis16/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/train-no-last-frame.json',
        NUM_CLASSES: 2,
        IMAGE_EXTENSION: '.jpg',
        IS_FLOW: False
    },
    'davis16_rgb_val': {
        IM_DIR: _DATA_DIR + '/davis16/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/val-no-last-frame.json',
        NUM_CLASSES: 2,
        IMAGE_EXTENSION: '.jpg',
        IS_FLOW: False
    },
    # DAVIS 16 datasets ending in _instance contain instance annotations from
    # DAVIS 2017, as opposed to the default DAVIS 2016 annotations, which are
    # binary masks.
    'davis16_flownet2_train_instance': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/2017-annotations/train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis16_flownet2_val_instance': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/2017-annotations/val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis16_rgb_train_instance': {
        IM_DIR: _DATA_DIR + '/davis16/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/2017-annotations/train-no-last-frame.json',
        NUM_CLASSES: 2,
        IMAGE_EXTENSION: '.jpg',
        IS_FLOW: False
    },
    'davis16_rgb_val_instance': {
        IM_DIR: _DATA_DIR + '/davis16/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/2017-annotations/val-no-last-frame.json',
        NUM_CLASSES: 2,
        IMAGE_EXTENSION: '.jpg',
        IS_FLOW: False
    },
    'davis16_liteflownet_val_instance': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis16/annotations/2017-annotations/val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
})

# YTVOS
#   ytvos_{input}_{split}
#
#   Split can be one of:
#   - sub_train_8-21-18
#       - Subset of YTVOS training set used for training.
#         TODO: Remove the date from this dataset.
#   - sub_val_8-21-18
#       - Subset of YTVOS training set used for evaluation.
#         TODO: Remove the date from this dataset.
#   - all_moving_sub_{train/val}_8-21-18
#       - Subset of sub_train/sub_val where all moving objects are labeled
#         and all static objects are not labeled.
#       - The date in this and all following datasets was the date when I
#         created this subset; it was a quick way to version the dataset,
#         since I had not filtered all of YTVOS yet.
#   - all_moving_strict_sub_{train/val}_8-21-18
#       - Subset of all_moving_sub_{train/val}_8-21-18, where we remove any
#         sequences where a large part of the object never moves, or objects
#         are over/undersegmented
#   - all_moving_strict_interesting_sub_{train/val}_8-21-18
#       - Subset of all_moving_strict_sub_{train/val}_8-21-18, created for
#         diagnostic purposes. Contains "interesting" objects, by a subjective
#         decision from me (@achald); these are usually objects that aren't in
#         standard detection datasets, like MS COCO.
#
# Path mapping:
#   /ytvos/train-splits/ ->
#       /data/achald/track/ytvos/train-splits/
#   /ytvos/all-moving-8-21-18/ ->
#       /data/achald/track/ytvos/moving-only/labels-8-21-18/all-moving/
DATASETS.update({
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
    'ytvos_hed_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/hed/train',
        ANN_FN: _DATA_DIR + '/ytvos/train-splits/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_hed_all_moving_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/hed/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_hed_all_moving_strict_sub_train_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/hed/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_hed_all_moving_strict_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/hed/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.png'
    },
    'ytvos_hed_all_moving_strict_interesting_sub_val_8-21-18': {
        IM_DIR: _DATA_DIR + '/ytvos/hed/train',
        ANN_FN: _DATA_DIR + '/ytvos/all-moving-8-21-18/strict/interesting-only/sub-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
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
})

for dataset, info in DATASETS.items():
    # We use "+" to indicate the concatenation of datasets internally.
    assert '+' not in dataset, '+ is an invalid character for datasets'
    if IMAGE_EXTENSION not in info:
        info[IMAGE_EXTENSION] = None
