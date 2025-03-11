import datetime
import cv2
import numpy as np

def draw_detection_boundary(frame, roi_start, width):
    """Draw a horizontal line showing the vehicle detection boundary."""
    cv2.line(frame, (0, roi_start), (width, roi_start), (0, 255, 0), 2)
    cv2.putText(frame, "Detection Zone", (width - 150, roi_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def create_detection_history_image(detection_history, width=1200, height=800, selected_lane=0):
    """
    Create an image showing recent detection history with payment status.
    Latest detections will appear at the bottom of the table.
    
    Args:
        detection_history: List of detection records
        width: Width of the history image (increased from 900 to 1200)
        height: Height of the history image (increased from 600 to 800)
        selected_lane: Lane to filter by (0 for all lanes)
        
    Returns:
        An image containing the detection history
    """
    # Create a blank image with white background
    history_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Header dimensions with increased height to prevent overlap
    header_height = 90
    row_height = 50
    
    # Column widths - adjusted for larger window and adding toll fee column
    time_col_width = 240
    type_col_width = 180
    lane_col_width = 100
    id_col_width = 100
    toll_fee_col_width = 100  # New column for toll fee
    status_col_width = width - time_col_width - type_col_width - lane_col_width - id_col_width - toll_fee_col_width - 40
    
    # Draw a rounded rectangle for the main content area - lighter color for white theme
    content_img = history_image.copy()
    cv2.rectangle(content_img, 
                 (15, header_height-10), 
                 (width-15, height-15), 
                 (240, 240, 245), -1)
    
    # Apply rounded corners effect using alpha blending
    alpha = 0.95
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (20, header_height-5), (width-20, height-20), 255, -1, cv2.LINE_AA)
    cv2.addWeighted(content_img, alpha, history_image, 1-alpha, 0, history_image)
    
    # Lane selection buttons with modern styling - positioned lower to avoid overlap with title
    button_width = 80
    button_height = 32
    button_y = header_height - 60
    button_margin = 10
    button_x = 30
    
    # Help text with subtle styling - positioned above buttons
    instruction_text = "Filter by lane:"
    cv2.putText(history_image, instruction_text, 
               (button_x, button_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
    
    # "All Lanes" button
    button_color = (220, 240, 220) if selected_lane == 0 else (240, 240, 240)
    button_border = (20, 120, 40) if selected_lane == 0 else (180, 180, 180)
    button_text_color = (20, 80, 20) if selected_lane == 0 else (80, 80, 80)
    
    cv2.rectangle(history_image, 
                 (button_x, button_y), 
                 (button_x + 100, button_y + button_height), 
                 button_color, -1, cv2.LINE_AA)
    
    # Button border
    cv2.rectangle(history_image, 
                 (button_x, button_y), 
                 (button_x + 100, button_y + button_height), 
                 button_border, 1, cv2.LINE_AA)
    cv2.putText(history_image, "All Lanes", 
               (button_x + 15, button_y + 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, button_text_color, 1, cv2.LINE_AA)
    button_x += 100 + button_margin
    
    # Lane-specific buttons
    for lane in range(1, 6):
        button_color = (220, 240, 220) if selected_lane == lane else (240, 240, 240)
        button_border = (20, 120, 40) if selected_lane == lane else (180, 180, 180)
        button_text_color = (20, 80, 20) if selected_lane == lane else (80, 80, 80)
        
        cv2.rectangle(history_image, 
                     (button_x, button_y), 
                     (button_x + button_width, button_y + button_height), 
                     button_color, -1, cv2.LINE_AA)
        
        # Button border
        cv2.rectangle(history_image, 
                     (button_x, button_y), 
                     (button_x + button_width, button_y + button_height), 
                     button_border, 1, cv2.LINE_AA)
        cv2.putText(history_image, f"Lane {lane}", 
                   (button_x + 15, button_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, button_text_color, 1, cv2.LINE_AA)
        button_x += button_width + button_margin
    
    # Add keyboard shortcut instructions - positioned well to the right for better spacing
    cv2.putText(history_image, "Press keys 0-5 to filter by lane", 
               (width - 300, button_y + 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1, cv2.LINE_AA)
    
    # Draw modern header row with light gradient for white theme
    header_img = history_image.copy()
    for y in range(header_height, header_height + row_height):
        gradient_factor = (y - header_height) / row_height
        color = (
            int(245 - gradient_factor * 20),
            int(245 - gradient_factor * 20),
            int(250 - gradient_factor * 10)
        )
        cv2.line(header_img, (20, y), (width - 20, y), color, 1)
    
    cv2.addWeighted(header_img, 0.7, history_image, 0.3, 0, history_image)
    
    # Draw column headers with improved typography - darker color for white background
    col_x = 20
    headers = ["Detection Time", "Vehicle Type", "Lane", "ID", "Toll Fee", "Payment Status"]
    col_widths = [time_col_width, type_col_width, lane_col_width, id_col_width, toll_fee_col_width, status_col_width]
    
    # Draw separator line under headers with gradient
    separator_img = history_image.copy()
    for x in range(20, width-20):
        gradient_factor = (x - 20) / (width - 40)
        color = (
            int(200 - gradient_factor * 50),
            int(200 - gradient_factor * 50),
            int(220 - gradient_factor * 50)
        )
        cv2.line(separator_img, 
                (x, header_height + row_height), 
                (x, header_height + row_height), 
                color, 1)
    cv2.addWeighted(separator_img, 0.7, history_image, 0.3, 0, history_image)
    
    # Make header background more visible against white
    cv2.rectangle(history_image, 
                 (20, header_height), 
                 (width - 20, header_height + row_height), 
                 (230, 235, 245), -1)
    
    for i, header in enumerate(headers):
        # Add subtle shadow effect to header text
        cv2.putText(history_image, header, 
                   (col_x + 10, header_height + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 80), 1, cv2.LINE_AA)
        
        # Draw subtle vertical separators between columns
        if i > 0:
            separator_img = history_image.copy()
            for y in range(header_height, height-20):
                gradient_factor = (y - header_height) / (height - header_height - 20)
                alpha = 0.3 - 0.2 * gradient_factor
                cv2.line(separator_img, 
                        (col_x, y), 
                        (col_x, y), 
                        (200, 200, 220), 1)
            cv2.addWeighted(separator_img, alpha, history_image, 1-alpha, 0, history_image)
        
        col_x += col_widths[i]
    
    # Filter records by lane if a specific lane is selected
    filtered_history = []
    if selected_lane == 0:
        filtered_history = detection_history.copy()
    else:
        filtered_history = [record for record in detection_history if record['lane'] == selected_lane]
    
    # Calculate how many rows we can show - more with increased height
    max_rows = min(len(filtered_history), (height - header_height - row_height - 20) // row_height)
    
    if max_rows == 0:
        # Show a message when no records match the filter
        message = f"No records found for Lane {selected_lane}"
        cv2.putText(history_image, message, 
                   (width // 2 - 150, height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 120), 1, cv2.LINE_AA)
        return history_image
    
    # Draw the data rows with improved styling for white theme
    start_index = max(0, len(filtered_history) - max_rows)
    
    for i in range(max_rows):
        record_index = start_index + i
        record = filtered_history[record_index]
        row_y = header_height + row_height + i * row_height
        
        # Draw alternating row backgrounds with subtle gradient for white theme
        if i % 2 == 0:
            row_img = history_image.copy()
            for y in range(row_y, row_y + row_height):
                gradient_factor = (y - row_y) / row_height
                color = (
                    int(240 - gradient_factor * 10),
                    int(240 - gradient_factor * 10),
                    int(245 - gradient_factor * 10)
                )
                cv2.line(row_img, (21, y), (width - 21, y), color, 1)
            cv2.addWeighted(row_img, 0.7, history_image, 0.3, 0, history_image)
        
        # Draw record data with improved styling for white theme
        col_x = 20
        
        # Time with subtle time-ago indicator
        time_text = record['time']
        cv2.putText(history_image, time_text, 
                   (col_x + 10, row_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 100), 1, cv2.LINE_AA)
        col_x += time_col_width
        
        # Vehicle type with improved color coding for white background
        vehicle_type = record['vehicle_type']
        if "2-wheel" in vehicle_type:
            type_color = (30, 150, 50)  # Green for 2-wheel
            bg_color = (220, 245, 220)
        elif "4-wheel" in vehicle_type:
            type_color = (100, 70, 180)  # Purple for 4-wheel
            bg_color = (235, 225, 245)
        elif "6-wheel" in vehicle_type or "6-more-wheel" in vehicle_type:
            type_color = (30, 120, 150)  # Teal for larger vehicles
            bg_color = (220, 235, 245)
        else:
            type_color = (100, 100, 100)  # Default gray
            bg_color = (240, 240, 240)
        
        # Add colored tag background for vehicle type
        tag_width = len(vehicle_type) * 15 + 20
        
        # Draw pill-shaped background for tag
        cv2.rectangle(history_image, 
                    (col_x + 5, row_y + 15), 
                    (col_x + tag_width, row_y + 45), 
                    bg_color, -1, cv2.LINE_AA)
        
        # Border for tag
        cv2.rectangle(history_image, 
                    (col_x + 5, row_y + 15), 
                    (col_x + tag_width, row_y + 45), 
                    type_color, 1, cv2.LINE_AA)
        
        cv2.putText(history_image, vehicle_type, 
                   (col_x + 12, row_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_color, 1, cv2.LINE_AA)
        col_x += type_col_width
        
        # Lane with modern styling
        lane_text = f"Lane {record['lane']}"
        cv2.putText(history_image, lane_text, 
                   (col_x + 10, row_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 100), 1, cv2.LINE_AA)
        col_x += lane_col_width
        
        # ID with modern styling
        id_text = f"ID: {record['id']}"
        cv2.putText(history_image, id_text, 
                   (col_x + 10, row_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 100), 1, cv2.LINE_AA)
        col_x += id_col_width
        
        # Toll fee with modern styling
        toll_fee = record.get('toll_fee', 0)
        toll_text = f"{toll_fee} THB"
        cv2.putText(history_image, toll_text, 
                   (col_x + 10, row_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 100, 30), 1, cv2.LINE_AA)
        col_x += toll_fee_col_width
        
        # Payment status with modern status indicator
        status = record['payment_status']
        
        # Draw status indicator
        if "Waiting" in status:
            status_color = (150, 80, 30)  # Orange for waiting
            bg_color = (250, 235, 220)  
        elif "Paid" in status:
            status_color = (30, 150, 50)  # Green for paid
            bg_color = (220, 245, 220)
        else:
            status_color = (180, 40, 40)  # Red for other states
            bg_color = (250, 220, 220)
        
        # Create status tag with rounded corners
        tag_width = len(status) * 12 + 45
        
        # Draw pill-shaped background for status
        cv2.rectangle(history_image, 
                    (col_x + 5, row_y + 15), 
                    (col_x + tag_width, row_y + 45), 
                    bg_color, -1, cv2.LINE_AA)
        
        # Border for status tag
        cv2.rectangle(history_image, 
                    (col_x + 5, row_y + 15), 
                    (col_x + tag_width, row_y + 45), 
                    status_color, 1, cv2.LINE_AA)
        
        cv2.putText(history_image, status, 
                   (col_x + 12, row_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)
    
    return history_image

def display_detection_history_window(detection_history, selected_lane=0):
    """
    Create and display a separate window showing recent detection history with payment status.
    
    Args:
        detection_history: List of dictionaries containing detection information
        selected_lane: Lane to filter by (0 for all lanes)
    """
    # Create history image with larger dimensions
    history_image = create_detection_history_image(detection_history, 1200, 800, selected_lane)
    
    # Display in a separate window with a specific size
    cv2.namedWindow("Vehicle Detection History", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Detection History", 1200, 800)
    cv2.imshow("Vehicle Detection History", history_image)
    
    return selected_lane
