import math
import cv2
import cvzone
import numpy as np
import datetime
from trackers import VehicleTracker, Sort
from detectors import LaneDetector
from constants import CLASS_NAMES
from utils.display_utils import (
    draw_detection_boundary, 
    display_table_in_separate_window,
    display_detection_history_window
)

if __name__ == "__main__":
  cap = cv2.VideoCapture("input_videos/input_video.mp4")
  
  vehicle_tracker = VehicleTracker(model_path="models/vehicle_tracker_model.pt")
  
  first_frame = cap.read()[1]
  lanes_detector = LaneDetector(first_frame)
  lanes_detector.detect()
  
  tracker = Sort()
  
  # Add tracking variables
  active_vehicles = {}  # {id: {'type': class_name, 'lane': lane_number, 'first_detected': timestamp}}
  disappeared_vehicles = {}  # {lane_number: [(id, type), ...]}
  vehicle_classes = {}  # {id: class_name}
  detection_history = []  # List of detection records with timestamp and payment status
  selected_lane = 0  # 0 means show all lanes, 1-5 means filter by lane
  
  while cap.isOpened():
    status, frame = cap.read()
    
    if not status:
      break
    
    # Store previous frame's active IDs
    previous_active_ids = set(active_vehicles.keys())
    
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
    
    # Get current timestamp for new detections
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
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
    
    # Clear current active vehicles and update with current frame data
    current_active_vehicles = {}
    
    for track in trackers:
      x1, y1, x2, y2, id = track
      x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
      w, h = x2 - x1, y2 - y1 
      
      # Display vehicle ID
      cvzone.putTextRect(frame, f"ID: {id}", (x1, y1-20), 1, 1, offset=5)
      cvzone.cornerRect(frame, (x1, y1, w, h), 10, rt=1)
      
      # Determine vehicle class from closest detection
      vehicle_class = None
      center_x = (x1 + x2) // 2
      
      # Find the vehicle class
      min_distance = float('inf')
      for i, det in enumerate(detections):
        det_x1, det_y1, det_x2, det_y2, _ = det
        det_center_x = (det_x1 + det_x2) / 2
        det_center_y = (det_y1 + det_y2) / 2
        dist = ((center_x - det_center_x)**2 + ((y1+y2)/2 - det_center_y)**2)**0.5
        if dist < min_distance and dist < 50:  # 50 pixels threshold
          min_distance = dist
          if i < len(r.boxes):
            vehicle_class = CLASS_NAMES[int(r.boxes[i].cls[0])]
      
      if vehicle_class:
        vehicle_classes[id] = vehicle_class
      
      # Determine which lane the vehicle is in
      center_x = (x1 + x2) // 2
      vehicle_lane = lanes_detector.get_lane_number(center_x)
      
      # Update active vehicles dictionary, preserving first detection time
      if id in active_vehicles:
        first_detected = active_vehicles[id].get('first_detected', current_time)
      else:
        first_detected = current_time
        
      current_active_vehicles[id] = {
        'type': vehicle_classes.get(id, "Unknown"),
        'lane': vehicle_lane,
        'first_detected': first_detected
      }
    
    # Find disappeared vehicles
    disappeared_ids = previous_active_ids - set(current_active_vehicles.keys())
    
    # Record disappeared vehicles by lane and add to detection history
    for disappeared_id in disappeared_ids:
      if disappeared_id in active_vehicles:
        vehicle_info = active_vehicles[disappeared_id]
        lane_number = vehicle_info['lane']
        vehicle_type = vehicle_info['type']
        detection_time = vehicle_info.get('first_detected', current_time)
        
        # Add to lane-specific disappeared vehicles
        if lane_number not in disappeared_vehicles:
          disappeared_vehicles[lane_number] = []
        
        disappeared_vehicles[lane_number].append((disappeared_id, vehicle_type))
        
        # Add to detection history with payment status (append to the end instead of inserting at the beginning)
        detection_history.append({
          'id': disappeared_id,
          'vehicle_type': vehicle_type,
          'lane': lane_number,
          'time': detection_time,
          'payment_status': "Waiting for payment"
        })
        
    # Update active vehicles for next frame
    active_vehicles = current_active_vehicles
    
    # Display the table of disappeared vehicles in a separate window
    display_table_in_separate_window(disappeared_vehicles)
    
    # Display detection history in a separate window with lane filtering
    display_detection_history_window(detection_history, selected_lane)
    
    # Draw a horizontal line showing the detection boundary
    frame = draw_detection_boundary(frame, roi_start, width)
        
    cv2.imshow("Trollway detection", frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break
    elif key == ord("0"):
      selected_lane = 0  # Show all lanes
    elif key >= ord("1") and key <= ord("5"):
      selected_lane = key - ord("0")  # Show specific lane (1-5)
  
  cap.release()
  cv2.destroyAllWindows()