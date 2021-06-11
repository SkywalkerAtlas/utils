import random

import cv2
# import some common detectron2 utilities
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer


def visual_test(dataset_name, json_path, image_dir, num_of_sample=1):
    register_coco_instances(dataset_name, {}, json_path, image_dir)
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    for d in random.sample(dataset_dicts, num_of_sample):
        print(d["file_name"])
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('image', vis.get_image()[:, :, ::-1])
        cv2.waitKey()


if __name__ == '__main__':
    # images_folder = '/home/sky/data/SLP/simLab/combined/images'
    # annotations_folder = '/home/sky/data/SLP/simLab/combined/annotations'
    #
    # visual_test('SLP', '{}/annotations.json'.format(annotations_folder), images_folder, 1)

    json_path = '/home/sky/data/Thermal-pose_data_set/annotations/thermal_post_keypoints.json'
    rgb_path = '/home/sky/data/Thermal-pose_data_set/images/RGB_images'

    visual_test('thermal_post_train', json_path, rgb_path, 1)

    # images_folder = '/home/sky/data/coco/images/val2017'
    # annotations_folder = '/home/sky/data/coco/annotations'
    # visual_test('SLP', '{}/person_keypoints_val2017.json'.format(annotations_folder), images_folder, 1)
