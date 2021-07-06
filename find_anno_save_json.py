# Some basic setup:
# Setup detectron2 logger
import datetime
import glob

import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

from converts import convert_box, convert_keypoints, save_to_json, logger
# from visual import test_visual

config_path = '/home/sky/detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'
checkpoint_path = '/home/sky/checkpoint/model_final_5ad38f.pkl'
rgb_path = '/home/sky/data/Thermal-pose_data_set/images/RGB_images'
thermal_path = '/home/sky/data/Thermal-pose_data_set/images/Thermal_images'

kp_thresh = 0.2


def covert_output_to_anno(predictions):
    """ Covert detectron2 output['instances'] into suitable annotation format.
    :param predictions: output['instances']
    :return annotation:
    """
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    # If no boxes were detected, return None.
    if boxes is not None:
        boxes = convert_box(boxes)
        num_instances = len(boxes)
        # Covert XYXY to XYHW
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
    else:
        return None

    # if num_instances is greater than 3 or ==0, means there is something wrong with the detector, return None
    if num_instances > 3 or num_instances == 0:
        return None

    if keypoints is not None:
        if num_instances:
            assert len(keypoints) == num_instances
        else:
            num_instances = len(keypoints)
        keypoints = convert_keypoints(keypoints)

    return boxes, classes, keypoints


def get_anno_dict():
    """Get COCO annotations from input images.
    :return:
    """
    logger.info("Generating prediction results into COCO format")
    coco_annotations = []
    coco_images = []

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    cfg.MODEL.WEIGHTS = checkpoint_path

    predictor = DefaultPredictor(cfg)

    for file in tqdm(glob.glob('{}/*.jpg'.format(rgb_path))):
        im = cv2.imread(file)
        coco_image = {
            'id': file.rsplit('/')[-1].split('.')[0],  # Get image id from file name
            'width': im.shape[1],
            'height': im.shape[0],
            'file_name': file.rsplit('/')[-1],
        }
        coco_images.append(coco_image)

        outputs = predictor(im)
        res = covert_output_to_anno(outputs['instances'].to('cpu'))
        if res is None:
            continue
        else:
            boxes, classes, keypoints = res

        for box, category, keypoints_each in zip(boxes, classes, keypoints):
            # Create a new dict with COCO fields
            coco_annotation = {}

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation['bbox'] = [round(float(x), 3) for x in box]
            coco_annotation['id'] = len(coco_annotations) + 1
            coco_annotation['image_id'] = coco_image['id']
            coco_annotation['category_id'] = category.item() + 1

            # Keypoints
            keypoints_list = [kp.item() for kp in keypoints_each.reshape(51, 1)]
            # If detected keypoints confident score is less than kp threshold, set that keypoint to 0. Otherwise set keypoints visibility to 2 (visible).
            for idx in range(2, len(keypoints_list), 3):
                if keypoints_list[idx] < kp_thresh:
                    keypoints_list[idx - 2], keypoints_list[idx - 1], keypoints_list[idx] = 0, 0, 0
                else:
                    keypoints_list[idx] = 2
                    keypoints_list[idx - 2] = int(keypoints_list[idx - 2])
                    keypoints_list[idx - 1] = int(keypoints_list[idx - 1])

            coco_annotation['keypoints'] = keypoints_list
            coco_annotation['num_keypoints'] = sum(kp > 0 for kp in keypoints_list[2::3])
            coco_annotation["iscrowd"] = 0

            coco_annotations.append(coco_annotation)

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file in Thermal Images.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"supercategory": "person", "id": 1, "name": "person",
                        "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                                      "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                                      "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
                        "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                                     [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6],
                                     [5, 7]]}],
        "licenses": None,
    }

    return coco_dict


if __name__ == '__main__':
    logger = setup_logger(name=__name__)
    json_path = '/home/sky/data/Thermal-pose_data_set/annotations/thermal_post_keypoints.json'
    coco_anno = get_anno_dict()
    save_to_json(coco_anno, json_path)
