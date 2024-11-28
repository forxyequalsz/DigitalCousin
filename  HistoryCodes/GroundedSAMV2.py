from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import SamPredictor, sam_model_registry
import cv2
import torch
import numpy as np
import os

# 配置模型和图片路径
CONFIG_PATH = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "Grounded-Segment-Anything/GroundingDINO/weights/groundingdino_swint_ogc.pth"  # GroundingDINO权重文件
SAM_CHECKPOINT = "Grounded-Segment-Anything/SAM/weights/sam_vit_l_0b3195.pth"  # SAM权重文件
SAM_MODEL_TYPE = "vit_l"  # SAM模型类型，可选 "vit_h", "vit_l", "vit_b"
DEVICE = "cpu"  # 可选 "cuda" 或 "cpu"
IMAGE_PATH = "TestImage/testimage2.jpeg"  # 输入图片路径
TEXT_PROMPT = "spoon"  # 文本提示
OUTPUT_DIR = "SegmentImage"  # 输出目录

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载GroundingDINO模型和图片
image_source, image = load_image(IMAGE_PATH)  # 图片预处理
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)  # 加载GroundingDINO模型

# 执行预测（图片，文本）-> 获取目标框
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=0.35,  # 可调整边界框置信度阈值
    text_threshold=0.25,  # 可调整文本置信度阈值
    device=DEVICE,
)

# 标注图片并保存边界框结果
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite(os.path.join(OUTPUT_DIR, "annotated_image.jpg"), annotated_frame)

# 加载SAM模型
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# 转换GroundingDINO的box为适合SAM输入的格式
boxes = boxes[:, :4]  # 提取前四列（xyxy格式）
H, W = image.shape[:2]
boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)  # 确保x坐标在图片宽度范围内
boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)  # 确保y坐标在图片高度范围内

# 将图片输入SAM
predictor.set_image(image_source)

# 获取原图的宽度和高度
height, width, _ = image_source.shape

# 调试输出文件夹
DEBUG_DIR = "DebugOutput"
os.makedirs(DEBUG_DIR, exist_ok=True)

# 转换GroundingDINO的box为适合SAM输入的格式
boxes = boxes[:, :4]  # 提取前四列（xyxy格式）
H, W = image.shape[:2]
boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)  # 确保x坐标在图片宽度范围内
boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)  # 确保y坐标在图片高度范围内
# 将 boxes 转换为 NumPy 格式
boxes = boxes.cpu().numpy()

# 生成每个检测框对应的mask
masks = []
for i, box in enumerate(boxes):
    # 反归一化坐标
    xmin, ymin, xmax, ymax = box
    xmin = int(xmin * width)
    ymin = int(ymin * height)
    xmax = int(xmax * width)
    ymax = int(ymax * height)
    # 生成mask：确保将box转换为 numpy 数组
    box_np = np.array([xmin, ymin, xmax, ymax]).reshape(1, -1)

    # 生成mask
    mask, _, _ = predictor.predict(
        point_coords=None,  # 不使用点提示
        point_labels=None,
        box=box_np,  # 使用GroundingDINO的检测框作为提示
        multimask_output=False,  # 仅输出一个mask
    )
    
    # 检查结果
    # print(f"Box {i} (normalized): {box}")
    # print(f"Box {i} (pixel): [{xmin}, {ymin}, {xmax}, {ymax}]")
    print(f"Unique values in mask_{i}: {np.unique(mask)}")
    # print(f"Mask shape: {mask.shape}")

    # 检查生成的掩码维度
    if mask is None or mask.size == 0:
        print(f"No mask generated for Box {i}.")
        continue

    # 确保掩码为二维
    if len(mask.shape) == 3:  # 如果是 [N, H, W]
        mask = mask[0]  # 取第一张掩码

    print(f"Mask shape: {mask.shape}")

    # 将掩码调整为图像大小
    mask_resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    print(f"Mask shape: {mask.shape}")

    # 转换掩码为 uint8
    mask_resized = (mask_resized > 0).astype(np.uint8) * 255
    print(f"Unique values in resized mask_{i}: {np.unique(mask_resized)}")

    # 保存原始掩码
    cv2.imwrite(os.path.join(DEBUG_DIR, f"raw_mask_{i}.jpg"), mask_resized)

    # 叠加显示 mask
    mask_image = image_source.copy()
    mask_image[mask_resized == 255] = [0, 255, 0]  # 将掩码区域标为绿色
    cv2.imwrite(os.path.join(DEBUG_DIR, f"mask_{i}.jpg"), mask_image)

    masks.append(mask_resized)