import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def gen_color_encoding(src_image: np.ndarray, overlay_image: np.ndarray):
    max_scale = np.amax(src_image)
    mask = src_image == 0
    scale = int(255 / max_scale)
    src_image = cv2.applyColorMap(src_image * scale, colormap=cv2.COLORMAP_JET)
    src_image[mask, :] = [0, 0, 0]
    overlay_image = cv2.addWeighted(overlay_image, 0.7, src_image, 0.3, 0)
    return overlay_image


data_dir = "../../../TuSimple_lane_detection/"
files = os.listdir(data_dir + 'images')
file = files[50]
image = cv2.imread(data_dir + 'images/' + file)
label = cv2.imread(data_dir + 'ground_truth/' + file[:-3] + 'png')[..., 0]  # has each lane encoded as 0, 1, 2 etc.
print(np.unique(label))
label_map = gen_color_encoding(label, image)
cv2.imwrite("Figures/sample_visualization_2.png", label_map)
plt.imshow(label_map)
plt.show()
