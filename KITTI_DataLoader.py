import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mayavi import mlab


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
        """
        generates data from the given KITTI data directory
        :return: generator which returns dictionary containing (image, label, calib file, lidar points) every iteration
        """
        file_names = os.listdir(self.image_dir)
        for file in file_names:
            image = self.read_image(os.path.join(self.image_dir, file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            lidar = self.read_velodyne(os.path.join(self.lidar_dir, file[:-3] + 'bin'))
            calib = self.read_calib(os.path.join(self.calib_dir, file[:-3] + 'txt'))
            label = self.read_label(os.path.join(self.label_dir, file[:-3] + 'txt'))
            yield {'image': image, 'label': label, 'calib': calib, 'lidar': lidar}

    def visualize(self, num_samples: int):
        """
        Visualizes the read image, labels with bounding boxes and also projects velodyne points onto left camera image
        :param num_samples: number of samples to visualize
        :return: shows the visualization of samples as matplotlib plots
        """
        gen_data = self.generate_data()
        for i in range(num_samples):
            data = next(gen_data)
            image = np.copy(data['image'])
            for labels in data['label']:
                image = labels.mark_bbox_2d(image)
            P2, velo2cam = data['calib']
            valid_lidar = self.project_lidar_to_left_camera(data['lidar'], velo2cam, P2)
            # mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point')
            # mlab.show()
            plt.scatter(valid_lidar[:, 0].astype(np.int32), valid_lidar[:, 1].astype(np.int32), c=valid_lidar[:, 2],
                        cmap='rainbow_r', alpha=0.2, s=0.5)
            plt.imshow(image)
            plt.show()

    @staticmethod
    def read_image(image_path: str) -> np.ndarray:
        """
        reads a path and returns numpy array corresponding to image read
        :param image_path: str - path to image file
        :return: numpy array with shape (1241, 374, 3)
        """
        image = cv2.imread(image_path)
        return image

    @staticmethod
    def read_label(label_path: str) -> list:
        """
        reads the labels file and converts to a KittiObjectClass object which has all underlying values relevant to
        labels
        :param label_path: str - path to label file
        :return: list of KittiObjectClass objects whose length contains number of objects in the label.
        NB: "DontCare" labels are skipped
        """
        with open(label_path, 'r') as label_file:
            label = label_file.readlines()
        objects = [KittiObjectClass(obj) for obj in label]
        objects = [obj for obj in objects if obj.name != 'DontCare']
        return objects

    @staticmethod
    def read_calib(calib_file_path: str) -> tuple:
        """
        reads calibration file given in the KITTI dataset folder and return the Projection matrix of left camera and
        Translation/rotation matrix from velodyne LIDAR to Camera.
        :param calib_file_path: path to calibration file of the image
        :return: (Projection_matrix, Tr_velo_to_cam multiplied with R0_rect)
        Projection matrix shape: 3 x 4  -> maps camera 3D coordinate to camera 2D image coordinates
        Tr_velo_to_cam matrix shape: 4 x 4 -> maps velodyne 3D coordinates to camera 3D coordinates
        """
        calib_dict = {'P2': [], 'R0_rect': [], 'Tr_velo_to_cam': []}
        with open(calib_file_path, 'r') as file:
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
        :return: numpy array with velodyne observations (num_points, 4) -> (x, y, z, reflectance or intensity)
        """
        velodyne = np.fromfile(velodyne_path, dtype=np.float32).reshape((-1, 4))
        return velodyne

    @staticmethod
    def project_lidar_to_left_camera(lidar_points: np.ndarray, velo2cam_mat: np.ndarray, P2: np.ndarray):
        """
        converts lidar points (x, y, z, intensity) to image coordinates (u, v, intensity) using the homographic
        transformation operation. See http://www.cvlibs.net/publications/Geiger2013IJRR.pdf for details.
        Some parts of the code inspired from https://github.com/azureology/kitti-velo2cam/blob/master/proj_velo2cam.py
        :param lidar_points: numpy array with shape (num_points, 4)
        :param velo2cam_mat: matrix for conversion from velodyne coordinates to camera 3d coordinates
        :param P2: matrix to convert from camera 3D coordinate to image 2D coordinate
        :return: Filtered image coordinates adhering to the camera image dimensions (1241, 374)
        """
        assert lidar_points.shape[-1] == 4, print(f"Expected shape - (n_points, 4) but got {lidar_points.shape}")
        # lidar points have shape - (x -> (towards front of car), y, z intensity)
        max_row = 374  # dimension of 2D images - height
        max_col = 1241  # dimension of 2D images - Width
        mask = lidar_points[:, 0] > 0
        lidar_points = lidar_points[mask, :]
        lidar_copy = np.copy(lidar_points)
        lidar_copy[:, 3] = 1
        lidar_pts = np.matmul(lidar_copy, velo2cam_mat.T)
        image_pts = np.matmul(lidar_pts, P2.T)
        mask = image_pts[:, 2] > 0  # division by zero points
        image_pts = image_pts[mask, :]
        lidar_points = lidar_points[mask, :]
        x, y = image_pts[:, 0] / image_pts[:, 2], image_pts[:, 1] / image_pts[:, 2]
        mask = (x >= 0) * (x < max_col) * (y >= 0) * (y < max_row)
        x, y, z = x[mask], y[mask], lidar_points[:, 3][mask]
        image_pts = np.vstack([x, y, z]).T
        return image_pts


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
        self.xmin = int(data[4])  # 2D bounding box of object in the image (0-based index): contains left, top pixels
        self.ymin = int(data[5])  # 2D Bounding box bottom right pixel coordinates
        self.xmax = int(data[6])
        self.ymax = int(data[7])
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        self.dim_h = data[8]  # 2d object dimensions height, width, length in m
        self.dim_w = data[9]
        self.dim_l = data[10]
        self.loc = (data[11], data[12], data[13])  # 3D object location x,y,z in camera coordinates (in meters)
        self.rotation_y = data[14]  # Rotation ry around Y-axis in camera coordinates [-pi..pi]

    def mark_bbox_2d(self, image: np.ndarray) -> np.ndarray:
        image = cv2.rectangle(image, (self.xmin, self.ymin), (self.xmax, self.ymax), color=(0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        image = cv2.putText(image, text=self.name, org=(self.xmin, self.ymin), color=color, fontScale=fontScale,
                            thickness=None, fontFace=font)
        return image


data_dir = "../KITTI_3D/training/"
im_dir = data_dir + "image_2/"
lbl_dir = data_dir + "label_2/"
clb_dir = data_dir + "calib/"
lid_dir = data_dir + "velodyne/"
kitti = KITTI(im_dir, lbl_dir, lid_dir, clb_dir)
kitti.visualize(8)
