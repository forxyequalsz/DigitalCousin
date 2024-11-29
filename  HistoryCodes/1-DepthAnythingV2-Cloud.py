import os
import json
import numpy as np
from transformers import pipeline
from PIL import Image
import torch
import pyvista as pv

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
image_path = 'docs/test_images/DSC_3752.JPG'
image = Image.open(image_path)

# 如果图像尺寸过大，进行缩放
max_width = 1200
max_height = 1200
image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

# 执行深度估计
print("Estimating depth...")
depth_result = pipe(image)

# 提取深度图 tensor（返回的结果包含一个深度图 tensor）
depth_tensor = depth_result['predicted_depth']
depth_array = depth_tensor.cpu().numpy()  # 将其转换为 NumPy 数组（确保是 CPU 上）

# 归一化深度图到 0-255 范围，并转换为 uint8 类型（可视化用）
depth_image = Image.fromarray((255 * (depth_array / np.max(depth_array))).astype(np.uint8))

# 保存深度图
depth_image.save("depth_image.png")
print("Depth image saved successfully.")

# 加载相机内参矩阵（假设文件为 camera_caliparams.json）
with open('camera_caliparams.json', 'r') as f:
    params = json.load(f)
camera_matrix = np.array(params['camera_matrix'])

# 使用 PyVista 生成点云
print("Generating point cloud...")

# 创建网格的坐标（x, y, z）坐标
height, width = depth_array.shape
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
xx = xx.flatten()
yy = yy.flatten()
zz = depth_array.flatten()

# 将深度图像的每个像素坐标转换为 3D 坐标（假设深度单位为毫米）
# 需要使用相机内参矩阵
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

# 使用相机内参进行 2D 图像坐标到 3D 空间的投影
X = (xx - cx) * zz / fx
Y = (yy - cy) * zz / fy
Z = zz  # 深度值即为 Z 坐标

# 创建点云数据
points = np.vstack((X, Y, Z)).T  # 组合为 [N x 3] 点坐标数组

# 创建 PyVista 点云对象
point_cloud = pv.PolyData(points)

# 将深度值（zz）添加到点数据中
point_cloud.point_data["depth"] = zz  # 使用 point_data 而不是 point_arrays

# 保存点云为 PLY 文件
point_cloud.save("output_point_cloud.ply")
print("Point cloud saved successfully.")

# 可视化点云
print("Visualizing point cloud...")
point_cloud.plot(render_points_as_spheres=True, point_size=5, color="white")