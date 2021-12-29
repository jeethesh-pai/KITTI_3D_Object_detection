import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt

class KITTI:
    def __init__(self, image_dir, label_dir, lidar_dir, calib_dir):
        self.image_dir = image_dir
        self.calib_dir = calib_dir
        self.label_dir = label_dir
        self.lidar_dir = lidar_dir
        assert len(os.listdir(self.image_dir)) == len(os.listdir(self.label_dir)), "No. of images != No. of Labels"
        self.num_samples = len(os.listdir(self.image_dir))

    def __len__(self):
        return self.num_samples

    def generate_data(self):
        file_names = os.listdir(self.image_dir)
        for file in file_names:
            image = self.read_image(os.path.join(self.image_dir, file))
            lidar = self.read_velodyne(os.path.join(self.lidar_dir, file[:-3] + 'bin'))
            calib = self.read_calib(os.path.join(self.calib_dir, file[:-3] + 'txt'))
            label = self.read_label(os.path.join(self.label_dir, file[:-3] + 'txt'))
            yield {'image': image, 'label': label, 'calib': calib, 'lidar': lidar}

    def visualize(self, num_samples: int):
        gen_data = self.generate_data()
        for i in range(num_samples):
            data = next(gen_data)

    @staticmethod
    def read_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        return image

    @staticmethod
    def read_label(label_path: str) -> list:
        with open(label_path, 'r') as label_file:
            label = label_file.readlines()
        objects = [KittiObjectClass(obj) for obj in label]
        return objects

    @staticmethod
    def read_calib(calib_file_path: str) -> tuple:
        """
        reads calibration file given in the KITTI dataset folder and return the Projection matrix of left camera and
        Translation/rotation matrix from velodyne LIDAR to Camera.
        :param calib_file_path: path to calibration file of the image
        :return: (Projection_matrix, Tr_velo_to_cam)
        """
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
        velo2cam_mat = np.matmul(calib_dict['R0_rect'], calib_dict['Tr_velo_to_cam'])
        return calib_dict['P2'], velo2cam_mat

    @staticmethod
    def read_velodyne(velodyne_path: str) -> np.ndarray:
        """
        reads the lidar bin files and gives it in ndarray format (entries, n_cols)
        n_cols are (x, y, z, intensity)
        :param velodyne_path:
        :return:
        """
        velodyne = np.fromfile(velodyne_path, dtype=np.float32).reshape((-1, 4))
        return velodyne


class KittiObjectClass:
    def __init__(self, label_file: str):
        data = label_file.split(sep=" ")
        data[1:] = [float(x) for x in data[1:]]
        self.name = data[0]  # 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or
        # 'DontCare'
        self.truncated = data[1]  # Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object
        # leaving image boundaries
        self.occlusion = int(data[2])  # 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
        self.alpha = data[3]  # Observation angle of object, ranging [-pi..pi]
        self.xmin = data[4]  # 2D bounding box of object in the image (0-based index): contains left, top pixels
        self.ymin = data[5]  # 2D Bounding box bottom right pixel coordinates
        self.xmax = data[6]
        self.ymax = data[7]
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        self.dim_h = data[8]  # 2d object dimensions height, width, length in m
        self.dim_w = data[9]
        self.dim_l = data[10]
        self.loc = (data[11], data[12], data[13])  # 3D object location x,y,z in camera coordinates (in meters)
        self.rotation_y = data[14]  # Rotation ry around Y-axis in camera coordinates [-pi..pi]


data_dir = "../KITTI_3D/training/label_2/000025.txt"
kitti = KITTI()
label_from = kitti.read_label(data_dir)
new_obj = [KittiObjectClass(obj) for obj in label_from]
print(label_from)
