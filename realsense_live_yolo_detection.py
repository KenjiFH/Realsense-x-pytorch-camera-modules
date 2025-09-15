import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

#model/pipeline setup from RS dependencies
yolo_model = YOLO('yolov5s.pt')  # this will be locally cached after downloading if it does not already exist as a .PT file

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        #get frames from pipeline
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy
        np_color_img = np.asanyarray(color_frame.get_data())

        # Run YOLO on NP array
        results = yolo_model(np_color_img)[0]

        # Draw bounding boxes for associated object
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            cv2.rectangle(np_color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(np_color_img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Realsense camera view with CV2
        cv2.imshow('RealSense view', np_color_img)
        #waitkey 1 for video 0 for image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
