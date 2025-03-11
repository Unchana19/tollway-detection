import math
import cv2
import cvzone
import numpy as np
from trackers import VehicleTracker, Sort
from detectors import LaneDetector
from constants import CLASS_NAMES
  
if __name__ == "__main__":
  cap = cv2.VideoCapture("input_videos/input_video.mp4")
  
  vehicle_tracker = VehicleTracker(model_path="models/vehicle_tracker_model.pt")
  
  first_frame = cap.read()[1]
  lanes_detector = LaneDetector(first_frame)
  lanes_detector.detect()
  
  tracker = Sort()
  
  while cap.isOpened():
    status, frame = cap.read()
    
    if not status:
      break
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Create a copy of the frame
    detection_frame = frame.copy()
    # Create a region of interest - bottom 2/3 of the frame
    roi_start = height // 3  # Start at 1/3 from the top
    roi = detection_frame[roi_start:height, 0:width]

    frame = lanes_detector.display_lane(frame)
       
    # Track vehicles only in the bottom 2/3
    results = vehicle_tracker.track(roi)
    detections = np.empty((0, 5))
    
    # Track vehicles
    for r in results:
      boxes = r.boxes
      for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Adjust y-coordinates to account for the ROI offset
        y1 = y1 + roi_start
        y2 = y2 + roi_start
        
        cls = int(box.cls[0])
        
        conf = math.ceil(box.conf[0] * 100) / 100
        if conf > 0.5:
          cvzone.putTextRect(frame, f"{CLASS_NAMES[cls]}", (x1, y2+20), 1, 1, offset=5)
          current_array = np.array([x1, y1, x2, y2, conf])
          detections = np.vstack([detections, current_array])
          
    trackers = tracker.update(detections)
    
    for track in trackers:
      x1, y1, x2, y2, id = track
      x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
      w, h = x2 - x1, y2 - y1 
      
      # Display vehicle ID
      cvzone.putTextRect(frame, f"ID: {id}", (x1, y1-20), 1, 1, offset=5)
      cvzone.cornerRect(frame, (x1, y1, w, h), 10, rt=1)
    
    # Draw a horizontal line showing the detection boundary
    cv2.line(frame, (0, roi_start), (width, roi_start), (0, 255, 0), 2)
        
    cv2.imshow("Trollway detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
  
  cap.release()
  cv2.destroyAllWindows()