from yacs.config import CfgNode as CN
_C = CN()
# -------------------------------------------------------- #
#                           Input                          #
# -------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.DATA_ROOT = "/Volumes/T7/fineGainImages/train"
_C.INPUT.EPOCH = 300
# Number of images per batch
_C.INPUT.BATCH_SIZE_TRAIN = 300
_C.INPUT.BATCH_SIZE_TEST = 300

# -------------------------------------------------------- #
#                           Miscs                          #
# -------------------------------------------------------- #
_C.BACKBONE = "resnet50"
_C.BACKBONE_PRETRAINED = True
# Directory where output files are written
_C.OUTPUT_DIR = "./output"


def get_default_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
