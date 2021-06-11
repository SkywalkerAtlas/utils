import os
from glob import glob
import re
import numpy as np
from scipy.io import loadmat
import shutil
import cv2
from converts import convert_box, convert_keypoints, save_to_json
from tqdm import tqdm
import json
import datetime
from collections import defaultdict

from detectron2.utils.logger import setup_logger
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from visual import visual_test
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
from fvcore.common.file_io import PathManager, file_lock
from detectron2.data.datasets import register_coco_instances

# def glob_re(pattern, strings):
#     return list(filter(re.compile(pattern).match, strings))


lsp2coco_mapping = {
    0: 16,
    1: 14,
    2: 12,
    3: 11,
    4: 13,
    5: 15,
    6: 10,
    7: 7,
    8: 5,
    9: 6,
    10: 8,
    11: 10,
    12: -1,
    13: -1,
}
lsp_num_of_keypoints = 14
coco_num_of_keypoints = 17
coco2lsp_mapping = {}
coco2lsp_mapping = defaultdict(lambda: -1, coco2lsp_mapping)

coco_indexs = list(range(coco_num_of_keypoints))

for key, value in lsp2coco_mapping.items():
    if value not in coco_indexs:
        continue
    else:
        coco2lsp_mapping[value] = key


def convert_lsp_to_coco(lsp_ndarray):
    """ Convert LSP dataset into COCO format, all the absence joints will be marked as 0 (no joints existed), it an suboptimal solution.
    :param lsp_ndarray: 1D ndarray, input keypoints in LSP format.
    :return coco_ndarray: 1D ndarray, converted keypoints in coco format.
    """
    coco_ndarray = np.zeros((3 * coco_num_of_keypoints))
    for i, lsp_idx in enumerate(range(0, len(lsp_ndarray) - 2, 3)):
        coco_idx = lsp2coco_mapping[i]
        # If there is no corresponding point in coco format, ignore the lsp keypoint
        if coco_idx == -1:
            continue
        # x and y indicate pixel positions in the image.
        # v indicates visibility— v=0: not labeled (in which case x=y=0),
        # v=1: labeled but not visible, and v=2: labeled and visible
        coco_x_idx, coco_y_idx, coco_v_idx = coco_idx * 3, coco_idx * 3 + 1, coco_idx * 3 + 2
        lsp_x_idx, lsp_y_idx, lsp_v_idx = lsp_idx, lsp_idx + 1, lsp_idx + 2
        coco_ndarray[coco_x_idx] = lsp_ndarray[lsp_x_idx]
        coco_ndarray[coco_y_idx] = lsp_ndarray[lsp_y_idx]
        coco_ndarray[coco_v_idx] = 2 if (lsp_ndarray[lsp_x_idx] > 0 and lsp_ndarray[lsp_y_idx] > 0) else 0

    return coco_ndarray


def convert_mat_to_json(folder_path, modality):
    """Read keypoint annotations from .mat files in each subfolder and mapping them to ids.
    :param folder_path: root folder to the dataset (sim or lab)
    :param modality: str, 'RGB' or "IR'
    :return:
    """
    sub_folders = [f for f in sorted(glob(r'{}/*/'.format(folder_path))) if re.search(r'\d{5}', f)]
    mat_name_format = 'joints_gt_{}.mat'
    keypoints_anno = {}

    for folder_count, sub_folder in enumerate(sub_folders):
        annots = loadmat(os.path.join(sub_folder, mat_name_format.format(modality)))
        mat_keypoints = annots['joints_gt']
        mat_keypoints = np.transpose(mat_keypoints, [2, 1, 0])
        mat_keypoints = mat_keypoints.reshape(mat_keypoints.shape[0], -1)
        for i, kp in enumerate(mat_keypoints):
            counter = folder_count * len(mat_keypoints) + i + 1
            kp = convert_lsp_to_coco(kp)
            # Convert keypoints to int.
            kp = list(kp)
            kp = [int(v) for v in kp]
            keypoints_anno['{}'.format(f"{counter:0>6}")] = kp

    return keypoints_anno


def save_keypoints_json(save_path, keypoints_anno, modality='RGB'):
    """Save dic format keypoints for each folder into json file.
    :param save_path: str, path to the saved (output) json file.
    :param keypoints_anno: dictionary, the grouped keypoints from each folder.
    :param modality: str, 'RGB' or 'IR'

    :return: None
    """
    with open(os.path.join(save_path, '{}_keypoints_anno.json'.format(modality)), 'w') as output_file:
        json.dump(keypoints_anno, output_file)


def move_image_pairs(folder_path, cover_needed, modalities_needed, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sub_folders = [f for f in sorted(os.listdir(folder_path)) if re.search(r'\d{5}', f)]

    for folder_count, sub_folder in enumerate(sub_folders):
        for cover in cover_needed:
            for modalities in modalities_needed:
                sorted_files = sorted(glob(os.path.join(folder_path, sub_folder, modalities, cover, '*.png')))
                for i, file in enumerate(sorted_files):
                    counter = folder_count * len(sorted_files) + i + 1
                    shutil.copy(file, '{}/{}_{}_{}.png'.format(output_folder, cover, modalities, f"{counter:0>6}"))
                    # print('{}->{}'.format(file, '{}/{}_{}_{}.png'.format(output_folder, cover, modalities, f"{counter:0>6}")))


def get_bbox(predictor, im):
    outputs = predictor(im)
    outputs = outputs['instances'].to('cpu')
    boxes = outputs.pred_boxes if outputs.has("pred_boxes") else None
    scores = outputs.scores if outputs.has("scores") else None

    # If no boxes were detected, return None.
    if boxes is not None:
        boxes = convert_box(boxes)
        num_instances = len(boxes)
    else:
        return None

    # if num_instances is 0, means there is something wrong with the detector, return None
    if num_instances == 0:
        return None
    # if num_instance is greater than 1, return the bounding box with largest scores
    if num_instances == 1:
        box = boxes
    else:
        box = boxes[np.argmax(scores.numpy())].reshape(1, -1)

    # Covert XYXY to XYHW
    box[:, 2] -= box[:, 0]
    box[:, 3] -= box[:, 1]
    return box


def get_anno_dict(images_folder, checkpoint_path, keypoint_json_path):
    """Get COCO annotations from input images.
    :param images_folder: str, path to the image folder.
    :param checkpoint_path: str, path to the pre-trained bounding box detector.
    :param keypoint_json_path: str, path to the keypoint annotation json file.
    :return coco_dict: dict, the formatted dict to save to json file.
    """

    logger.info("Generating prediction results into COCO format")
    coco_annotations = []
    coco_images = []

    # Prepare bounding box detector
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
    cfg.MODEL.WEIGHTS = checkpoint_path

    predictor = DefaultPredictor(cfg)

    # Prepare keypoint annotations.
    with open(keypoint_json_path) as f:
        keypoints_annos = json.load(f)

    for file in tqdm(glob('{}/*.png'.format(images_folder))):
        im = cv2.imread(file)
        coco_image = {
            'id': file.rsplit('/')[-1].split('.')[0],  # Get image id from file name
            'width': im.shape[1],
            'height': im.shape[0],
            'file_name': file.rsplit('/')[-1],
        }
        coco_images.append(coco_image)

        # Get the bounding box
        bbox = get_bbox(predictor, im)
        assert len(bbox) == 1, 'The number of detected bounding box should be 1, got {} instead.'.format(len(bbox))

        coco_annotation = {}
        # COCO requirement:
        #   linking annotations to images
        #   "id" field must start with 1
        coco_annotation['bbox'] = [round(float(x), 3) for x in bbox.reshape(-1)]
        coco_annotation['id'] = len(coco_annotations) + 1
        coco_annotation['image_id'] = coco_image['id']
        coco_annotation['category_id'] = 1
        # Keypoints
        coco_annotation['keypoints'] = keypoints_annos[coco_image['id'].rsplit('_', 1)[-1]]
        coco_annotation['num_keypoints'] = sum(kp > 0 for kp in coco_annotation['keypoints'][2::3])
        coco_annotation["iscrowd"] = 0

        coco_annotations.append(coco_annotation)

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file in SLP Images.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"supercategory": "person","id": 1,"name": "person","keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}],
        "licenses": None,
    }

    return coco_dict


if __name__ == '__main__':
    logger = setup_logger(name=__name__)
    create_new_image = False
    create_new_keypoint_json = False
    target_images_folder = '/home/sky/data/SLP/simLab/combined/images'
    annotations_folder = '/home/sky/data/SLP/simLab/combined/annotations'
    save_path = '/home/sky/data/SLP/simLab/combined/annotations'

    # checkpoint_path = '/home/sky/checkpoint/model_final_5ad38f.pkl'

    folder_path = '/home/sky/data/SLP/simLab'
    cover_needed = ['uncover']
    modalities_needed = ['RGB']

    # LSP_format_keypoints = read_mat(file_path)

    if create_new_image:
        move_image_pairs(folder_path, cover_needed, modalities_needed, target_images_folder)
    if create_new_keypoint_json:
        for modality in modalities_needed:
            keypoints_anno = convert_mat_to_json(folder_path, modality)
            save_keypoints_json(annotations_folder, keypoints_anno, modality='RGB')
    anno_dict = get_anno_dict(target_images_folder, checkpoint_path, '{}/RGB_keypoints_anno.json'.format(annotations_folder))
    save_to_json(anno_dict, '{}/annotations.json'.format(annotations_folder))
