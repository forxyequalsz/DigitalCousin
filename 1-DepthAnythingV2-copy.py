import os
import json
import numpy as np
from transformers import pipeline
from PIL import Image
import open3d as o3d
import torch

# 确保设备设置为 MPS（如果在 M1/M2 Mac 上）或 CPU
print("Checking device...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

# 使用 Open3D 生成点云
print("Generating point cloud...")

# 创建 Open3D 的深度图像对象（需要 uint16 类型）
depth_o3d = o3d.geometry.Image(depth_array.astype(np.uint16))

# 创建相机内参对象
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=depth_array.shape[1],
    height=depth_array.shape[0],
    fx=camera_matrix[0, 0],
    fy=camera_matrix[1, 1],
    cx=camera_matrix[0, 2],
    cy=camera_matrix[1, 2]
)

# 生成点云
point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
    depth_o3d, intrinsics, depth_scale=1000.0, depth_trunc=1000.0, stride=1
)

# 保存点云
o3d.io.write_point_cloud("output_point_cloud.ply", point_cloud)
print("Point cloud saved successfully.")

# 可视化点云
print("Visualizing point cloud...")
o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud Visualization")
