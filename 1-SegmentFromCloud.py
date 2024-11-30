import os
import numpy as np
import pyvista as pv
from PIL import Image
import json

# 加载相机内参
with open('camera_caliparams.json', 'r') as f:
    params = json.load(f)
camera_matrix = np.array(params['camera_matrix'])

# 加载未分割的点云图
point_cloud = pv.read('DepthImage/output_point_cloud.ply')

# 加载深度图
depth_image_path = 'DepthImage/depth_image.png'
depth_image = Image.open(depth_image_path)
depth_array = np.array(depth_image)

# 获取点云的所有点
points = point_cloud.points

# 获取原始深度图的高度和宽度
height, width = depth_array.shape

# 计算点云中每个点的像素坐标
fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
x_coords = ((points[:, 0] * fx) / points[:, 2]) + cx
y_coords = ((-points[:, 1] * fy) / points[:, 2]) + cy  # Y轴同样进行取反，保持一致

# 将 x, y 坐标转换为整数索引，注意需要限制在图像的尺寸范围内
x_indices = np.clip(x_coords.astype(np.int32), 0, width - 1)
y_indices = np.clip(y_coords.astype(np.int32), 0, height - 1)

# 遍历所有 mask 文件并可视化覆盖的点云部分
mask_dir = 'SegmentImage/Masks'
output_dir = 'SegmentImage/SegmentedPointClouds'
os.makedirs(output_dir, exist_ok=True)

for mask_file in os.listdir(mask_dir):
    if mask_file.endswith('.png'):
        # 加载 mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = Image.open(mask_path).convert('L')  # 确保是灰度图像
        mask_array = np.array(mask)

        # 使用 mask 对点云进行筛选，确保只选择该 mask 标记的区域
        mask_values = mask_array[y_indices, x_indices]
        filtered_points = points[(mask_values == 255)]

        # 创建新的 PyVista 点云对象
        filtered_point_cloud = pv.PolyData(filtered_points)

        # 保存分割后的点云
        filtered_point_cloud_path = os.path.join(output_dir, f"segmented_{mask_file.split('.')[0]}.ply")
        filtered_point_cloud.save(filtered_point_cloud_path)
        print(f"Segmented point cloud saved as '{filtered_point_cloud_path}'")

        # 可视化分割后的点云
        plotter = pv.Plotter()
        plotter.add_mesh(point_cloud, render_points_as_spheres=True, color='gray', point_size=3)  # 添加整体点云
        plotter.add_mesh(filtered_point_cloud, render_points_as_spheres=True, color='red', point_size=5)  # 添加分割的点云
        plotter.show_grid()

        # 调整相机视角
        plotter.camera_position = [(0, 0, 20), (0, 0, 0), (0, 1, 0)]
        plotter.show()  # 使用调整后的视角
        print(f"Visualized mask overlay for '{mask_file}'")