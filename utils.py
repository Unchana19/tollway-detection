import cv2
import numpy as np

def draw_lane_line(frame):
    """
    Enhanced function to detect and draw lane lines with improved accuracy
    Takes only frame as input and returns the frame with lane lines drawn
    """
    # Make a copy of the frame to avoid modifying the original
    result = frame.copy()
    height, width = frame.shape[:2]
    
    # 1. Apply color filtering (focus on white and yellow lanes)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # White color mask
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Yellow color mask
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)
    
    # 2. Convert to grayscale and apply Gaussian blur with optimized parameters
    gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny edge detection with optimized thresholds
    edges = cv2.Canny(blur, 50, 150)
    
    # 4. Define region of interest - focus on bottom half of the frame
    mask = np.zeros_like(edges)
    polygon = np.array([
        [(0, height), 
         (width, height), 
         (width * 0.55, height * 0.6), 
         (width * 0.45, height * 0.6)]
    ], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    
    # Apply the mask
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 5. Hough line detection with optimized parameters
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi/180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )
    
    # 6. Process and draw the lines
    if lines is not None:
        # Separate left and right lines based on slope
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope
            if x2 - x1 == 0:  # Avoid division by zero
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter lines based on slope
            if abs(slope) < 0.5:  # Ignore horizontal lines
                continue
            
            if slope < 0:  # Negative slope is left line
                left_lines.append(line[0])
            else:  # Positive slope is right line
                right_lines.append(line[0])
        
        # Function to average and extrapolate lines
        def get_averaged_line(lines, is_right=False):
            if not lines:
                return None
            
            x_sum, y_sum, m_sum = 0, 0, 0
            
            for line in lines:
                x1, y1, x2, y2 = line
                
                # Get slope and y-intercept
                m = (y2 - y1) / (x2 - x1) if (x2 != x1) else 0
                b = y1 - m * x1
                
                # Sum values
                m_sum += m
                x_sum += (x1 + x2) / 2
                y_sum += (y1 + y2) / 2
            
            # Calculate average values
            m_avg = m_sum / len(lines)
            x_avg = x_sum / len(lines)
            y_avg = y_sum / len(lines)
            b_avg = y_avg - m_avg * x_avg
            
            # Calculate endpoints for the averaged line
            y_bottom = height
            y_top = int(height * 0.6)  # Top point aligned with region of interest
            
            x_bottom = int((y_bottom - b_avg) / m_avg) if m_avg != 0 else 0
            x_top = int((y_top - b_avg) / m_avg) if m_avg != 0 else 0
            
            return [x_bottom, y_bottom, x_top, y_top]
        
        # Get averaged lines
        left_line = get_averaged_line(left_lines)
        right_line = get_averaged_line(right_lines, is_right=True)
        
        # Draw lines
        line_thickness = 10
        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
        
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
    
    return result
