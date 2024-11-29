import os
import json
import torch
import cv2
import numpy as np
import pyvista as pv
from transformers import pipeline
from PIL import Image
from libs.resize_image import resize_image

# 确保设备设置为 MPS（如果在 M1/M2 Mac 上）或 CPU
print("Checking device...")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# 设置 Hugging Face 模型中心的镜像站点
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载深度估计模型（Depth-Anything-V2-Small-hf）
print("Loading depth estimation model...")
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

# 打开图像并调整大小（避免图像过大，导致计算时间过长）
IMAGE_PATH = 'docs/test_images/DSC_3752.JPG'
resized_image = resize_image(IMAGE_PATH)
# 保存缩放后图片
resized_image_path = "docs/test_images/resized_image.jpg"
cv2.imwrite(resized_image_path, resized_image)
print(f"图像进行缩放，并存储于 {resized_image_path}")

# 缩放后的图片路径赋值给image
image = resized_image_path

# 执行深度估计
print("Estimating depth...")
depth_result = pipe(image)

# 提取深度图 tensor（返回的结果包含一个深度图 tensor）
depth_tensor = depth_result['predicted_depth']
depth_array = depth_tensor.cpu().numpy()  # 将其转换为 NumPy 数组（确保是 CPU 上）

# 归一化深度图到 0-255 范围，并转换为 uint8 类型（可视化用）
depth_image = Image.fromarray((255 * (depth_array / np.max(depth_array))).astype(np.uint8))

# 保存深度图
depth_image.save("DepthImage/depth_image.png")
print("Depth image saved successfully.")

# 加载相机内参矩阵（假设文件为 camera_caliparams.json）
with open('camera_caliparams.json', 'r') as f:
    params = json.load(f)
camera_matrix = np.array(params['camera_matrix'])

# 生成点云
print("Generating point cloud...")
height, width = depth_array.shape
fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

points = []
for v in range(height):
    for u in range(width):
        z = depth_array[v, u]
        if z == 0:
            continue
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points.append([x, y, z])

points = np.array(points)

# 创建 PyVista 点云对象
point_cloud = pv.PolyData(points)

# 修正水平镜像问题
point_cloud.points[:, 0] *= -1

# 可视化点云图
print("Visualizing point cloud...")
# point_cloud.plot(render_points_as_spheres=True, point_size=5, color='gray')
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=5, color='gray')
plotter.show_grid()
plotter.camera_position = [(0, 0, 20), (0, 0, 0), (0, -1, 0)]
plotter.show()
print("Point cloud visualization complete.")

# 保存点云图
point_cloud.save('DepthImage/output_point_cloud.ply')
print("Point cloud saved successfully.")