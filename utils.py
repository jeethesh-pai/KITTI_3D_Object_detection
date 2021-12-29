import os
import re
import numpy as np

data_dir = "../KITTI_3D/training/calib/000025.txt"
calib_dict = {'P2': [], 'R0_rect': [], 'Tr_velo_to_cam': []}
with open(data_dir, 'r') as file:
    data = file.readlines()
for line in data:
    if line[:2] == 'P2':
        calib_dict['P2'] = np.array(re.split(" ", line[4:]), dtype=np.float32).reshape((3, 4))
    elif line[:7] == 'R0_rect':
        rect = np.array(re.split(" ", line[9:]), dtype=np.float32).reshape((3, 3))
        rect = np.concatenate([rect, [[0], [0], [0]]], axis=-1)
        rect = np.concatenate([rect, [[0, 0, 0, 1]]], axis=0)
        calib_dict['R0_rect'] = rect
    elif line[:14] == 'Tr_velo_to_cam':
        Tr_velo_to_cam = np.array(re.split(" ", line[16:]), dtype=np.float32).reshape((3, 4))
        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, [[0, 0, 0, 1]]], axis=0)
        calib_dict['Tr_velo_to_cam'] = Tr_velo_to_cam
print(calib_dict)