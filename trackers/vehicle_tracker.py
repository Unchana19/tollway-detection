from ultralytics import YOLO

class VehicleTracker:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    
  def track(self, image):
    return self.model(image, stream=True)