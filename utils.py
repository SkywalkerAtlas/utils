import re
import os
import cv2

import numpy as np
from scipy.io import loadmat
from glob import glob


def read_mat(folder_path, modality):
    """Read keypoints from mat file.

    Args:
        folder_path ([str]): file path to the destination folder.
        modality ([str]): modality to extract from. Chosen from ('RGB', 'IR')

    Returns:
        keypoints ([ndarray]): ndarray with shape (N_sample, N_keypoints, 3)
    """
    sub_folders = [f for f in sorted(glob(r'{}/*/'.format(folder_path))) if re.search(r'\d{5}', f)]
    mat_name = 'joints_gt_{}.mat'.format(modality)

    for folder_count, sub_folder in enumerate(sub_folders):
        annots = loadmat(os.path.join(sub_folder, mat_name))
        keypoints = annots['joints_gt']
        keypoints = np.transpose(keypoints, [2, 1, 0])
        
    return keypoints


def align(im1, im2, kp1, kp2, method=cv2.RANSAC):
    """ Align image 1 with image 2 via homograph matrix.

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


def crop_with_image(im1, im2):
    """Crop image 1 to the size of image 2, the size of image 1 should be lager than the size of image 2.

    Args:
        im1 (numpy.ndarray): source image with shape (h1, w1)
        im2 (numpy.ndarray): target image with shape (h2, w2), h2 < h1, w2 < w1.

    Returns:
        numpy.ndarray: the cropped image with shape (h2, w2)
    """
    assert im2.shape[1] <= im1.shape[1] and im2.shape[0] <= im1.shape[0], 'im1 should be larger than im2, got im1: {}, im2: {} instead'.format(im1.shape[:2], im2.shape[:2])
    return im1[:im2.shape[0], :im2.shape[1], :]