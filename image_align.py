import cv2

from utils import read_mat, align, crop_to_image

if __name__ == '__main__':
    folder_path = '/Users/skywalker/Downloads/simLab'
    rgb_keypoints = read_mat(folder_path, 'RGB')
    ir_keypoints = read_mat(folder_path, 'IR')

    im1 = cv2.imread('/Users/skywalker/Downloads/simLab/00001/RGB/uncover/image_000008.png')
    im2 = cv2.imread('/Users/skywalker/Downloads/simLab/00001/IR/uncover/image_000008.png')

    kp1 = rgb_keypoints[7, :, :]
    kp2 = ir_keypoints[7, :, :]

    aligned_im1 = align(im1, im2, kp1, kp2)
    aligned_im1 = crop_to_image(aligned_im1, im2)

    cv2.imwrite('rgb.png', aligned_im1)
    cv2.imwrite('ir.png', im2)

    # cv2.imshow('img', aligned_im1)
    # cv2.waitKey(0)