import numpy as np
import json
import os

import numpy as np
# import some common detectron2 utilities
from detectron2.structures import Boxes, Keypoints, RotatedBoxes
from detectron2.utils.logger import setup_logger
from fvcore.common.file_io import PathManager, file_lock

logger = setup_logger(name=__name__)


def convert_box(boxes):
    if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
        return boxes.tensor.numpy()
    else:
        return np.asarray(boxes)


def convert_keypoints(keypoints):
    if isinstance(keypoints, Keypoints):
        keypoints = keypoints.tensor
    keypoints = np.asarray(keypoints)
    return keypoints


def save_to_json(coco_dict, output_file, allow_cached=True):
    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of data to COCO format ...)")
            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            with PathManager.open(output_file, "w") as f:
                json.dump(coco_dict, f)
