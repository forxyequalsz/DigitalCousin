import os
import json
import numpy as np
import pyvista as pv
import cv2

# 1. 加载原始相机内参矩阵
with open('camera_caliparams.json', 'r') as f:
    params = json.load(f)
camera_matrix = np.array(params['camera_matrix'])
fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

# 2. 加载原始深度图和缩放信息
original_image_size = (6000, 4000)  # 假设原始图像大小
depth_image_path = "DepthImage/depth_image.png"
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
height, width = depth_image.shape

# 3. 调整相机内参矩阵
scale_x = width / original_image_size[0]
scale_y = height / original_image_size[1]

fx_new = fx * scale_x
fy_new = fy * scale_y
cx_new = cx * scale_x
cy_new = cy * scale_y

# 更新内参矩阵
adjusted_camera_matrix = np.array([
    [fx_new, 0, cx_new],
    [0, fy_new, cy_new],
    [0, 0, 1]
])

# 4. 生成全局点云
points = []
for v in range(height):
    for u in range(width):
        z = depth_image[v, u]
        if z == 0:
            continue
        # 根据调整后的相机内参矩阵进行像素坐标到相机坐标系的转换
        x = (u - cx_new) * z / fx_new
        y = -(v - cy_new) * z / fy_new  # 注意这里对 y 取负
        points.append([x, y, z])

points = np.array(points)
global_point_cloud = pv.PolyData(points)

# 5. 加载分割 mask
mask_path = "SegmentImage/Masks/mask_1.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 6. 分割全局点云中的物体
segmented_points = []
for v in range(height):
    for u in range(width):
        if mask[v, u] > 0:  # 如果该像素属于物体
            z = depth_image[v, u]
            if z == 0:
                continue
            x = (u - cx_new) * z / fx_new
            y = -(v - cy_new) * z / fy_new
            segmented_points.append([x, y, z])

segmented_points = np.array(segmented_points)
segmented_point_cloud = pv.PolyData(segmented_points)

# 7. 估算物体的大小
x_min, y_min, z_min = segmented_points.min(axis=0)
x_max, y_max, z_max = segmented_points.max(axis=0)
length = x_max - x_min
width = y_max - y_min
height = z_max - z_min

print(f"Estimated object dimensions (Length, Width, Height): {length:.2f}m, {width:.2f}m, {height:.2f}m")

# 8. 可视化点云
plotter = pv.Plotter()
plotter.add_mesh(global_point_cloud, render_points_as_spheres=True, point_size=5, color='gray')
plotter.add_mesh(segmented_point_cloud, render_points_as_spheres=True, point_size=5, color='red')
plotter.show_grid()
plotter.camera_position = [(0, 0, 20), (0, 0, 0), (0, 1, 0)]
plotter.show()