import os
from glob import glob
import re
from utils import align
import numpy as np
from scipy.io import loadmat
import shutil
import cv2
import itertools
from converts import convert_box, convert_keypoints, save_to_json
from utils import read_mat, align, crop_to_image, convert_mat_to_json, convert_lsp_to_coco, save_keypoints_json
from tqdm import tqdm
import json
import random
import datetime
from collections import defaultdict

from detectron2.utils.logger import setup_logger
# import some common detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from visual import visual_test
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
# from fvcore.common.file_io import PathManager, file_lock
# from detectron2.data.datasets import register_coco_instances

# def glob_re(pattern, strings):
#     return list(filter(re.compile(pattern).match, strings))


def move_image_pairs(folder_path, cover_needed, output_folder, align_rgb2ir=True):
    """Move image pairs (IR-RGB) into a combined folder for following training. The directory layout should be the same with COCO dataset.

    Args:
        folder_path (str): The root directory of the original SLP dataset.
        cover_needed (list): The cover mode included into final dataset.
        output_folder (str): The output folder.
        align_rgb2ir (bool, optional): Option to align rgb images to ir images. Defaults to True.
    """
    if not os.path.exists(os.path.join(output_folder, 'IR')):
        os.makedirs(os.path.join(output_folder, 'IR'))
    if not os.path.exists(os.path.join(output_folder, 'RGB')):
        os.makedirs(os.path.join(output_folder, 'RGB'))
    if not os.path.exists(os.path.join(output_folder, 'alignedRGB')):
        os.makedirs(os.path.join(output_folder, 'alignedRGB'))

    counter = 0
    for lab in tqdm([f for f in sorted(os.listdir(folder_path)) if re.search(r'.*Lab', f)]):
        sub_folders = [f for f in sorted(os.listdir(os.path.join(folder_path, lab))) if re.search(r'\d{5}', f)]
        for sub_folder in tqdm(sub_folders):
            keypoints_rgb = read_mat(os.path.join(folder_path, lab, sub_folder), 'RGB') if align_rgb2ir else None
            keypoints_ir = read_mat(os.path.join(folder_path, lab, sub_folder), 'IR') if align_rgb2ir else None
            for cover in tqdm(cover_needed, leave=False):
                sorted_irs = sorted(glob(os.path.join(folder_path, lab, sub_folder, 'IR', cover, '*.png')))
                sorted_rgbs = sorted(glob(os.path.join(folder_path, lab, sub_folder, 'RGB', cover, '*.png')))
                for file_rgb, file_ir in zip(sorted_rgbs, sorted_irs):
                    tar_path_rgb = '{}/{}/{}.png'.format(output_folder, 'alignedRGB', f"{counter:0>6}")
                    tar_path_ir = '{}/{}/{}.png'.format(output_folder, 'IR', f"{counter:0>6}")
                    if align_rgb2ir:
                        im_rgb = cv2.imread(file_rgb)
                        im_ir = cv2.imread(file_ir)
                        kp_rgb = keypoints_rgb[counter % (len(sorted_rgbs)), :, :]
                        kp_ir = keypoints_ir[counter % (len(sorted_irs)), :, :]

                        im_rgb = crop_to_image(
                            im1=align(im_rgb, im_ir, kp_rgb, kp_ir),
                            im2=im_ir
                        )
                        cv2.imwrite(tar_path_rgb, im_rgb)
                        cv2.imwrite(tar_path_ir, im_ir)
                    else:
                        shutil.copy(file_rgb, '{}/{}/{}.png'.format(output_folder, 'RGB', f"{counter:0>6}"))
                        shutil.copy(file_ir, '{}/{}/{}.png'.format(output_folder, 'IR', f"{counter:0>6}"))
                        # print('{}->{}'.format(file_rgb, '{}/{}/{}.png'.format(output_folder, 'RGB', f"{counter:0>6}")))
                        # print('{}->{}'.format(file_ir, '{}/{}/{}.png'.format(output_folder, 'IR', f"{counter:0>6}")))
                        
                    counter += 1


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


def get_dummy_bbox(im):
    """Return dummy bounding box with image size.

    Args:
        im (numpy.ndarray): input image

    Returns:
        numpy.ndarray: dummy bounding box, take whole image as bounding box.
    """
    h, w = im.shape[:2]
    return np.array([0, 0, w, h]).reshape(1, -1)


def get_annotations_and_coco_image(images_folder, keypoint_json_path):
    """Get all COCO annotations for coco_annotations and coco_images.

    Args:
        images_folder (str): Filepath to the IR images.
        keypoint_json_path (str): Filepath to the presaved keypoint json file.

    Returns:
        coco_annotations (list): list of dicts, contains all the keypoint annotations for SLP datasets, the bbox is a dummy one. Will be filled into coco_annotations.
        coco_images (list): list of dicts, contains all the sub information, will be filled into coco_images.
    """

    logger.info("Generating prediction results into COCO format")
    coco_annotations = []
    coco_images = []

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
        # bbox = get_bbox(predictor, im)
        bbox = get_dummy_bbox(im)
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

    return coco_annotations, coco_images


def get_splitted_dataset(images_folder, keypoint_json_path, poportions=(0.9, 0.1, 0.0)):
    """Get COCO annotations for training and testing from input images.

    Args:
        images_folder (str): Filepath to the IR images.
        keypoint_json_path (str): Filepath to the presaved keypoint json file.
        poportion (tuple): Poportion of training_samples, val_samples and test_samples

    Returns:
        coco_dict_train (dict): coco formatted dict of training set, will be saved into json file.
        coco_dict_val (dict): coco formatted dict of validation set, will be saved into json file.
        coco_dict_test (dict): coco formatted dict of test set, will be saved into json file.
    """
    logger.info("Generating prediction results into COCO format")

    coco_annotations, coco_images = get_annotations_and_coco_image(images_folder, keypoint_json_path)
    assert len(coco_images) == len(coco_annotations), 'The length of coco_images and coco_annotations should be the same, got {}, {} instead'.format(
        len(coco_images), 
        len(coco_annotations)
        )

    # Generate indicators for three different dataset according to their poportions (CDF).
    cdf = tuple(itertools.accumulate(poportions))
    assert len(cdf) == 3, 'The length of poportions should be 3: (train, val, test), got {} instead'.format(len(cdf))
    assert cdf[-1] == 1.0, 'The cdf of poportions should be up to 1, got {} instead'.format(cdf[-1])

    probabilities = [random.random() for _ in range(len(coco_images))]
    indicators = ['train' if p < cdf[0] else 'val' if p < cdf[1] else 'test' for p in probabilities]

    train_selector = [True if indicator == 'train' else False for indicator in indicators]
    val_selector = [True if indicator == 'val' else False for indicator in indicators]
    test_selector = [True if indicator == 'test' else False for indicator in indicators]

    coco_images_train = list(itertools.compress(coco_images, train_selector))
    coco_annotations_train = list(itertools.compress(coco_annotations, train_selector))
    coco_images_val = list(itertools.compress(coco_images, val_selector))
    coco_annotations_val = list(itertools.compress(coco_annotations, val_selector))
    coco_images_test = list(itertools.compress(coco_images, test_selector))
    coco_annotations_test = list(itertools.compress(coco_annotations, test_selector))

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file in SLP Images.",
    }
    coco_dict_train = {
        "info": info,
        "images": coco_images_train,
        "annotations": coco_annotations_train,
        "categories": [{"supercategory": "person","id": 1,"name": "person","keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}],
        "licenses": None,
    }
    coco_dict_val = {
        "info": info,
        "images": coco_images_val,
        "annotations": coco_annotations_val,
        "categories": [{"supercategory": "person","id": 1,"name": "person","keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}],
        "licenses": None,
    }
    coco_dict_test = {
        "info": info,
        "images": coco_images_test,
        "annotations": coco_annotations_test,
        "categories": [{"supercategory": "person","id": 1,"name": "person","keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}],
        "licenses": None,
    }

    return coco_dict_train, coco_dict_val, coco_dict_test


def move_images_into_dir(anno_dict, src, des):
    """Move images into separate dir

    Args:
        anno_dict ([type]): [description]
        src ([type]): [description]
        des ([type]): [description]
    """
    if not os.path.exists(des):
        os.makedirs(des)
    logger.info('move files {} -> {}'.format(src, des))
    for image_info in anno_dict['images']:
        file_name = image_info['file_name']
        shutil.move(
            os.path.join(src, file_name),
            os.path.join(des, file_name)
        )


if __name__ == '__main__':
    logger = setup_logger(name=__name__)
    create_new_image = True
    create_new_keypoint_json = True
    align_rgb2ir = True
    target_images_folder = '/Users/skywalker/Downloads/SLP/combined/images'
    annotations_folder = '/Users/skywalker/Downloads/SLP/combined/annotations'
    save_path = '/Users/skywalker/Downloads/SLP/combined/annotations'

    checkpoint_path = '/home/sky/checkpoint/model_final_5ad38f.pkl'

    folder_path = '/Users/skywalker/Downloads/SLP'
    cover_conditions = ['uncover', 'cover1', 'cover2']
    modalities = ['RGB', 'IR']

    # LSP_format_keypoints = read_mat(file_path)
    if create_new_image:
        move_image_pairs(folder_path, cover_conditions, target_images_folder, align_rgb2ir)
    if create_new_keypoint_json:
        keypoints_anno = convert_mat_to_json(folder_path)
        save_keypoints_json(annotations_folder, keypoints_anno, modality='IR')
    anno_dict_train, anno_dict_val, anno_dict_test = get_splitted_dataset(
        os.path.join(target_images_folder, 'IR'),
        os.path.join(annotations_folder, '{}_keypoints_anno.json'.format('IR')),
        poportions=(0.9, 0.1, 0.0)
    )
    save_to_json(anno_dict_train, '{}/train_annotations.json'.format(annotations_folder))
    save_to_json(anno_dict_test, '{}/test_annotations.json'.format(annotations_folder))
    save_to_json(anno_dict_val, '{}/val_annotations.json'.format(annotations_folder))

    move_images_into_dir(anno_dict_train, os.path.join(target_images_folder, 'IR'), os.path.join(target_images_folder, 'IR', 'train'))
    move_images_into_dir(anno_dict_train, os.path.join(target_images_folder, 'alignedRGB'), os.path.join(target_images_folder, 'alignedRGB', 'train'))
    move_images_into_dir(anno_dict_val, os.path.join(target_images_folder, 'IR'), os.path.join(target_images_folder, 'IR', 'val'))
    move_images_into_dir(anno_dict_val, os.path.join(target_images_folder, 'alignedRGB'), os.path.join(target_images_folder, 'alignedRGB', 'val'))
    move_images_into_dir(anno_dict_test, os.path.join(target_images_folder, 'IR'), os.path.join(target_images_folder, 'IR', 'test'))
    move_images_into_dir(anno_dict_test, os.path.join(target_images_folder, 'alignedRGB'), os.path.join(target_images_folder, 'alignedRGB', 'test'))
