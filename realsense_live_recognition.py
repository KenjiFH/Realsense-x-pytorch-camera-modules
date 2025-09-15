import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision import datasets

# Load pretrained image recognition model
model = resnet18(pretrained=True)
model.eval()

# Set up transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Start RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Show RGB and depth
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense RGB + Depth', images)

        # Image recognition on RGB frame
        input_tensor = transform(color_image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output).item()
        print("Predicted class index:", predicted_class)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
