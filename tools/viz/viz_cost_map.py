import heapq

import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import pandas as pd

from neural_reconstruction.core.topology import TopologyBuilder
from neural_reconstruction.core.pathfinding import PathFinder
from neural_reconstruction.core.preprocessing import dilate_epidermis_vertically
from neural_reconstruction.algorithms.annotation_grow.dijkstra import get_components
from neural_reconstruction.algorithms.annotation_grow.graph_builder import (
    find_meeting_points,
    build_component_graph,
    prune_edges,
    minimum_spanning_forest,
)
from neural_reconstruction.algorithms.annotation_grow.skeleton import (
    build_result_graph,
)
from skimage.measure import regionprops
import matplotlib.cm as mcm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import skimage as ski

# from skimage.measure import label
# from skimage.restoration import rolling_ball
from scipy.spatial import KDTree
from neural_reconstruction.core.evaluation import (
    extract_graph_points,
    compute_average_hausdorff_distance,
    compute_point_min_distances,
)
from neural_reconstruction.core.crosses_detection import (
    SegmentDetector,
    RegionLabeler,
    CrossingCounter,
)

BASE_PATH = Path("/home/pony/projects/ienf_q/")
IMAGE_ID = "S222-2_a"
BASE_PATH = BASE_PATH / f"data_0331/{IMAGE_ID}"

image = np.array(cv2.imread(f"{BASE_PATH}/image.png", cv2.IMREAD_COLOR_RGB))

Image.fromarray(image[666 : 666 + 200, 4700 : 4700 + 200]).save(
    f"./output/viz_input_crop.png"
)

image = image[:, :, 1]  # 只取綠色通道
Image.fromarray(image).save(f"./output/viz_original.png")
Image.fromarray(image[666 : 666 + 200, 4700 : 4700 + 200]).save(
    f"./output/viz_original_crop.png"
)
mask = cv2.imread(f"{BASE_PATH}/mask.png", cv2.IMREAD_GRAYSCALE)
annotation = cv2.imread(f"{BASE_PATH}/weka.png", cv2.IMREAD_GRAYSCALE)
label_img = cv2.imread(f"{BASE_PATH}/label.png", cv2.IMREAD_GRAYSCALE)

roi_mask = dilate_epidermis_vertically(mask, offset_px=50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# background = ski.restoration.rolling_ball(image, radius=50)

image = cv2.subtract(image, background)
Image.fromarray(background).save(f"./output/viz_bg_only.png")
Image.fromarray(image).save(f"./output/viz_bg.png")
Image.fromarray(image[666 : 666 + 200, 4700 : 4700 + 200]).save(
    f"./output/viz_bg_sb_crop.png"
)
tileGridSize = 16
clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(tileGridSize, tileGridSize))
image = clahe.apply(image)

viz_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
orig_mask_bin = mask > 127
grown_bin = (roi_mask > 0) & ~orig_mask_bin
tint = viz_img.copy()
tint[orig_mask_bin] = (255, 255, 0)
tint[grown_bin] = (0, 255, 0)
viz_img = cv2.addWeighted(tint, 0.45, viz_img, 0.55, 0)

image = cv2.bitwise_and(image, image, mask=roi_mask)
image = ski.filters.sato(image, sigmas=range(3, 8), black_ridges=False)
image = (image - image.min()) / (image.max() - image.min()) * 255
image = image.astype(np.uint8)
cost_map = np.exp(1.0 - (image.astype(np.float32) / 255.0)) - 1.0

fig_x = 64
fig_y = (cost_map.shape[0] / cost_map.shape[1]) * fig_x
fig, axes = plt.subplots(1, 1, figsize=(fig_x, fig_y))
im = axes.imshow(cost_map)
axes.axis("off")

# add colorbar aligned with image height
divider = make_axes_locatable(axes)
cax = divider.append_axes("right", size="2%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=48)

plt.tight_layout()
plt.savefig(f"./output/viz_cost_map.png", dpi=300)
