import re
import os
import cv2

import json
import numpy as np
from scipy.io import loadmat
from glob import glob
from collections import defaultdict


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


def read_mat(folder_path, modality):
    """Read keypoints from mat file.

    Args:
        folder_path (str): file path to the destination folder.
        modality (str): modality to extract from. Chosen from ('RGB', 'IR')

    Returns:
        keypoints (ndarray): ndarray with shape (N_sample, N_keypoints, 3)
    """
    mat_name = 'joints_gt_{}.mat'.format(modality)

    annots = loadmat(os.path.join(folder_path, mat_name))
    keypoints = annots['joints_gt']
    keypoints = np.transpose(keypoints, [2, 1, 0])
        
    return keypoints


def align(im1, im2, kp1, kp2, method=cv2.RANSAC):
    """ Align image 1 to image 2 via homograph matrix.

    Args:
        im1 (numpy.ndarray): Image 1 with shape (h1, w1), the source image.
        im2 (numpy.ndarray): Image 2 with shape (h2, w2), the target image
        kp1 (numpy.ndarray): keypoint 1 with shape (num_keypoints, 3), the source keypoints.
        kp2 (numpy.ndarray): keypoint 2 with shape (num_keypoints, 3), the target keypoints.
        method (optional): method pass to cv2.findHomography. Defaults to cv2.RANSAC.

    Returns:
        aligned_im1: The aligned image 1. The keypoints of it should be aligned with image 2.
    """
    (H, mask) = cv2.findHomography(kp1, kp2, method=method)
    (h, w) = im1.shape[:2]
    aligned_im1 = cv2.warpPerspective(im1, H, (h, w))

    return aligned_im1


def crop_to_image(im1, im2):
    """Crop image 1 to the size of image 2, the size of image 1 should be lager than the size of image 2.

    Args:
        im1 (numpy.ndarray): source image with shape (h1, w1)
        im2 (numpy.ndarray): target image with shape (h2, w2), h2 < h1, w2 < w1.

    Returns:
        numpy.ndarray: the cropped image with shape (h2, w2)
    """
    assert im2.shape[1] <= im1.shape[1] and im2.shape[0] <= im1.shape[0], 'im1 should be larger than im2, got im1: {}, im2: {} instead'.format(im1.shape[:2], im2.shape[:2])
    return im1[:im2.shape[0], :im2.shape[1], :]


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


def convert_mat_to_json(folder_path, cover_conditions=['uncover', 'cover1', 'cover2'], modality='IR'):
    """Read keypoint annotations from .mat files in each subfolder and mapping them to ids.

    Args:
        folder_path (str): The root directory of the original SLP dataset.
        cover_conditions (list): The list for selected cover conditions, default ['uncover', 'cover1', 'cover2'], select all conditions.
        modality (str): Get keypoints of this modality, 'IR' or 'RGB', default 'IR'

    Returns:
        dict: key - counter_num, value - list of keypoints
    """
    labs = [f for f in sorted(os.listdir(folder_path)) if re.search(r'.*Lab', f)]
    mat_name = 'joints_gt_{}.mat'.format(modality)
    keypoints_anno = {}

    counter = 0
    for lab in labs:
        sub_folders = [f for f in sorted(os.listdir(os.path.join(folder_path, lab))) if re.search(r'\d{5}', f)]
        for sub_folder in sub_folders:
            annots = loadmat(os.path.join(folder_path, lab, sub_folder, mat_name.format(modality)))
            mat_keypoints = annots['joints_gt']
            mat_keypoints = np.transpose(mat_keypoints, [2, 1, 0])
            mat_keypoints = mat_keypoints.reshape(mat_keypoints.shape[0], -1)
            for _ in cover_conditions:
                for kp in mat_keypoints:
                    kp = convert_lsp_to_coco(kp)
                    # Convert all keypoints to int.
                    kp = list(kp)
                    kp = [int(v) for v in kp]
                    keypoints_anno[int('{}'.format(f"{counter:0>6}"))] = kp
                    counter += 1

    return keypoints_anno


def save_keypoints_json(save_path, keypoints_anno, modality='IR'):
    """Save dic format keypoints for each folder into json file.
    :param save_path: str, path to the saved (output) json file.
    :param keypoints_anno: dictionary, the grouped keypoints from each folder.
    :param modality: str, 'RGB' or 'IR'

    :return: None
    """
    with open(os.path.join(save_path, '{}_keypoints_anno.json'.format(modality)), 'w') as output_file:
        json.dump(keypoints_anno, output_file)