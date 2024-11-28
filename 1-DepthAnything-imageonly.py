import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import pipeline
from PIL import Image
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open('docs/test_images/testimage3.JPG')
depth = pipe(image)["depth"]
depth.save("DepthImage/depth.png")
print(depth)