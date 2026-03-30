from ultralytics import YOLO

# Load the official 'Nano' model (best for web speed)
model = YOLO('yolov8n.pt') 

# Export to ONNX format specifically for web use
# This creates a file named 'yolov8n.onnx'
model.export(format='onnx', imgsz=640)