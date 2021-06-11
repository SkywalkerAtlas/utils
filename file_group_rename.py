import os
import glob
import shutil

data_path = '/home/sky/data/Thermal-pose_data_set/'
des_rgb = '/home/sky/data/Thermal-pose_data_set/images/RGB_images'
des_thermal = '/home/sky/data/Thermal-pose_data_set/images/Thermal_images'

dir_path_list = [f.path for f in os.scandir(data_path) if f.is_dir() and 'Data_set_person_label' not in str(f)]

counter = 0

for dir_path in dir_path_list:
    rgb_file_path = '{}/RGB_images'.format(dir_path)
    thermal_file_path = '{}/Thermal_images'.format(dir_path)

    for rgb_file, thermal_file in zip(sorted(glob.glob(r'{}/*.jpg'.format(rgb_file_path))), sorted(glob.glob(r'{}/*.jpg'.format(thermal_file_path)))):
        shutil.copy(rgb_file, '{}/{}.jpg'.format(des_rgb, f"{counter:0>6}"))
        shutil.copy(thermal_file, '{}/{}.jpg'.format(des_thermal, f"{counter:0>6}"))
        # print(rgb_file)
        # print(thermal_file)
        counter += 1
