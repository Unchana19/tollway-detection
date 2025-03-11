import cv2
import numpy as np

class LaneDetector:
  def __init__(self, frame):
    self.frame = frame
    self.lanes = []
    self.lane_colors = {}
    self.lane_spaces = []
    
  def detect(self):
    detect_lanes = []
    
    gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    edges = cv2.Canny(gray, 50, 70)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=300, maxLineGap=250)

    if lines is not None:
      grouped_lines = []
      for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Make sure y1 is always at the bottom (larger y value) and y2 at the top (smaller y value)
        if y1 < y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        if x2 != x1:
          angle = np.degrees(np.arctan((y2-y1)/(x2-x1)))
        else:
          angle = 90
        
        if abs(angle) > 50:
          found_group = False
          for group in grouped_lines:
            group_x1, group_y1, group_x2, group_y2 = group[0]
            group_angle = np.degrees(np.arctan((group_y2-group_y1)/(group_x2-group_x1))) if group_x2 != group_x1 else 90
            
            if abs(angle - group_angle) < 15 and np.sqrt((x1-group_x1)**2 + (y1-group_y1)**2) < 100:
              group.append([x1, y1, x2, y2])  # Store the sorted coordinates
              found_group = True
              break
          
          if not found_group:
            grouped_lines.append([[x1, y1, x2, y2]])  # Store the sorted coordinates
      
      # Store the grouped lines as lanes
      for group in grouped_lines:
        detect_lanes.append([group[0]])
        
      self.lanes = detect_lanes
      self.lanes.sort(key=lambda lane: lane[0][0])  # Sort by x1 value
      
      self.lane_spaces = []
      
      # Leftmost lane (from left edge of image to first line)
      self.lane_spaces.append({
          'lane_number': 1,
          'left_line': None,
          'right_line': self.lanes[0]
      })
      
      # Middle lanes (between detected lines)
      for i in range(len(self.lanes)-1):
          lane_number = i + 2
          left_line = self.lanes[i]
          right_line = self.lanes[i+1]
          self.lane_spaces.append({
              'lane_number': lane_number,
              'left_line': left_line,
              'right_line': right_line
          })
          
      # Rightmost lane (from last line to right edge of image)
      self.lane_spaces.append({
          'lane_number': len(self.lanes) + 1,
          'left_line': self.lanes[-1],
          'right_line': None
      })
    else:
      print("No lines detected")
      
  # Update to the display_lane method
  def display_lane(self, frame):
      result_frame = frame.copy()
      
      # Draw all lane lines first
      for i, lane in enumerate(self.lanes):
          line = lane[0]  # Get the first line in the lane
          x1, y1, x2, y2 = line
          
          # Use different colors for each line
          color = (0, 255, 0)  # Default green color
          
          # Draw the lane line
          cv2.line(result_frame, (x1, y1), (x2, y2), color, 2)
      
      # Then draw lane numbers in the center of each lane
      for lane_space in self.lane_spaces:
          lane_number = lane_space['lane_number']
          left_line = lane_space['left_line']
          right_line = lane_space['right_line']
          
          # Calculate center point of the lane
          if left_line is None:
              # Leftmost lane (edge of frame to first line)
              left_x = 0
              right_x = right_line[0][0]
          elif right_line is None:
              # Rightmost lane (last line to edge of frame)
              left_x = left_line[0][0]
              right_x = frame.shape[1]  # Width of the frame
          else:
              # Regular lanes between two lines
              left_x = left_line[0][0]
              right_x = right_line[0][0]
              
          center_x = int((left_x + right_x) / 2)
          center_y = int(frame.shape[0] * 0.95)
          
          # Draw lane number
          text = f"{lane_number}"
          cv2.putText(
              result_frame, 
              text, 
              (center_x-5, center_y), 
              cv2.FONT_HERSHEY_SIMPLEX, 
              1,  # Font scale
              (0, 0, 255),  # Red color (BGR)
              2,  # Thickness
              cv2.LINE_AA
          )
      
      return result_frame
      
  def get_lane_number(self, x_position):
      """Determine which lane a point is in based on its x-coordinate."""
      for lane_space in self.lane_spaces:
          lane_number = lane_space['lane_number']
          left_line = lane_space['left_line']
          right_line = lane_space['right_line']
          
          # Calculate lane boundaries
          if left_line is None:
              # Leftmost lane
              left_x = 0
              right_x = right_line[0][0]
          elif right_line is None:
              # Rightmost lane
              left_x = left_line[0][0]
              right_x = self.frame.shape[1]
          else:
              # Regular lanes between two lines
              left_x = left_line[0][0]
              right_x = right_line[0][0]
          
          if left_x <= x_position <= right_x:
              return lane_number
              
      return 0  # Not in any recognized lane
