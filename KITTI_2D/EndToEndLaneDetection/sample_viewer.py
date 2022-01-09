import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def gen_color_encoding(src_image: np.ndarray, overlay_image: np.ndarray):
    """
    Generates color map for the label image
    :param src_image: 2D numpy array with label mapping in the form [0, 1, 2, ...] corresponding to lane number
    :param overlay_image: image on to which color mapped image needs to be overlaid
    :return: color mapped image overlaid on to given label image for better visualization
    """
    max_scale = np.amax(src_image)
    mask = src_image == 0
    scale = int(255 / max_scale)
    src_image = cv2.applyColorMap(src_image * scale, colormap=cv2.COLORMAP_JET)
    src_image[mask, :] = [0, 0, 0]
    overlay_image = cv2.addWeighted(overlay_image, 0.7, src_image, 0.3, 0)
    return overlay_image


data_dir = "../../../TuSimple_lane_detection/"
files = os.listdir(data_dir + 'images')
for file in files[:10]:
    image = cv2.imread(data_dir + 'images/' + file)
    label = cv2.imread(data_dir + 'ground_truth/' + file[:-3] + 'png')[..., 0]  # label has each lane encoded as 1, 2...
    label_map = gen_color_encoding(label, image)
    # cv2.imwrite("Figures/sample_visualization_2.png", label_map) #  uncomment for writing sample image
    plt.imshow(label_map)
    plt.show()
