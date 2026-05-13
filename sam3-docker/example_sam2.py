from ultralytics import SAM
import cv2
from PIL import Image

model = SAM('sam2.pt')
image_path = 'data/test.jpg'
results = model(image_path)

for r in results:
    r.save('output/result.jpg')
    print("处理完成，结果已保存到 output/result.jpg")
