# 文件夹及功能说明

## 实现

* 1-CameraCalibration.py：负责基于OpenCV生成相机校准矩阵。
* 1-DepthAnythingV2.py：负责基于Depth-Anything-V2生成深度图。
* 1-GroundedSAMV2.py：负责基于prompt（稍作修改就可以通过GPT的API调用自动获取prompt），分割出指定物体的mask。

## 输出

* DepthImage：存储1-DepthAnythingV2.py输出的深度图和点云图。
* SegmentImage：存储1-GroundedSAMV2.py输出的masks和叠加的物体识别结果图。

## 其他资源

* weight：存放groundingDINO和SAM的预训练权重文件。
* docs：资源库，目前包括用于校准相机的棋盘格图、校准用的图片和用于测试的图片。
* camera_caliparams.json：存储的相机校准矩阵。
* HistoryCodes：存储了尝试过程中（多数是半对半错的）的code

## 安装外部依赖

* Depth-Anything-V2：一种基于深度学习的深度图生成器。可以在不依赖深度相机的情况下，通过深度估计直接生成深度图。
* Grounded-Segment-Anything：即GroundedSAM。一种视觉-语言模型技术，基于GroundingDINO和SegmentAnything共同实现，前者通过文本提示（prompts）识别图像中的特定物体或区域负责识别与定位，后者负责分割并生成mask。
  * pip install git+https://github.com/IDEA-Research/GroundingDINO
  * pip install git+https://github.com/facebookresearch/segment-anything
