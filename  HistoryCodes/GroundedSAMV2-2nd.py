import cv2
import torch
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops
from segment_anything import build_sam, SamPredictor
from PIL import Image

# 配置路径
CONFIG_PATH = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "Grounded-Segment-Anything/GroundingDINO/weights/groundingdino_swint_ogc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "TestImage/testimage2.jpeg"
TEXT_PROMPT = "spoon"

# 加载Grounding DINO模型和图片
image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

# 执行预测：得到边界框和文本标签
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=0.35,
    text_threshold=0.25,
    device=DEVICE,
)

# 标注图片
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# 加载SAM模型
sam_checkpoint = 'Grounded-Segment-Anything/SAM/weights/sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# 设置图像
sam_predictor.set_image(image_source)

# 反归一化坐标：转换为[xyxy]格式
H, W, _ = image_source.shape
boxes = torch.tensor(boxes)
# 将cx, cy, w, h转换为xmin, ymin, xmax, ymax
# boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])  # 将[0,1]范围的框坐标转换为像素坐标

# 转换为SAM可以使用的框
transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)

# 获取掩码
masks, _, _ = sam_predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=True
)

print(f"Masks shape: {masks.shape}")
print(f"Masks unique values: {torch.unique(masks)}")

# 显示掩码（可以选择随机颜色或者固定颜色）
def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])  # 默认蓝色
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

# 将掩码叠加到图像上
annotated_frame_with_mask = show_mask(masks[0][0], annotated_frame)
Image.fromarray(annotated_frame_with_mask)

# 保存最终结果
cv2.imwrite("SegmentImage/annotated_image_with_mask.jpg", annotated_frame_with_mask)