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


if __name__ == '__main__':
    folder_path = '/Users/skywalker/Downloads/simLab'
    rgb_keypoints = read_mat(folder_path, 'RGB')
    ir_keypoints = read_mat(folder_path, 'IR')

    