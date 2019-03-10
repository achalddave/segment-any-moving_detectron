# Datasets that are not used for our final set of experiments.

_DATA_DIR = ''

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

# DAVIS unused for final experiments:
#   davis_{input}_moving_{split}:
#     - Sequences where all moving objects are labeled and all static objects
#       are unlabeled, where {split} corresponds to a split that I made-up
#       randomly (so the validation split can contain sequences from DAVIS
#       17's official train/val splits).
#
# Path mapping:
#   /davis/annotations-without-davis16/
#       /data/achald/track/DAVIS/2017/moving-only/without-davis16
#   /davis/annotations-always-moving/:
#       /data/achald/track/DAVIS/2017/moving-only/relabeled-always-moving/json-annotations/
UNUSED_DAVIS = {
    'davis_rgb_moving_val': {
        IM_DIR: _DATA_DIR + '/davis/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IMAGE_EXTENSION: '.jpg',
        IS_FLOW: False
    },
    'davis_rgb_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-test-no-last-frame.json',
        NUM_CLASSES: 2,
        IMAGE_EXTENSION: '.jpg',
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
    'davis_hed_moving_val': {
        IM_DIR: _DATA_DIR + '/davis/hed/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-val-no-last-frame.json',
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
    'davis_flow_moving_val': {
        IM_DIR: _DATA_DIR + '/davis/liteflownet/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-val-no-last-frame.json',
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
    'davis_flownet2_flow_moving_val': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-val-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_flownet2_flow_moving_test': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-test-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
    'davis_rgb_moving_without_davis16_all': {
        IM_DIR: _DATA_DIR + '/davis/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis/annotations-without-davis16/moving-without-davis16-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False,
        IMAGE_EXTENSION: '.jpg'
    },
    'davis_flownet2_moving_without_davis16_all': {
        IM_DIR: _DATA_DIR + '/davis/flownet2/',
        ANN_FN: _DATA_DIR + '/davis/annotations-without-davis16/moving-without-davis16-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: True
    },
}

UNUSED_DAVIS16 = {
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
        ANN_FN: _DATA_DIR + '/davis16/moving-annotations/train-no-last-frame.json',
        IMAGE_EXTENSION: '.jpg',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis16_rgb_moving_test': {
        IM_DIR: _DATA_DIR + '/davis16/JPEGImages/',
        ANN_FN: _DATA_DIR + '/davis16/moving-annotations/test-no-last-frame.json',
        IMAGE_EXTENSION: '.jpg',
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
    'davis_flownet2_vis_moving_train': {
        IM_DIR: _DATA_DIR + '/davis/flownet2-vis/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-train-no-last-frame.json',
        NUM_CLASSES: 2,
        IS_FLOW: False
    },
    'davis_flownet2_vis_moving_val': {
        IM_DIR: _DATA_DIR + '/davis/flownet2-vis/',
        ANN_FN: _DATA_DIR + '/davis/annotations/moving-val-no-last-frame.json',
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
}
