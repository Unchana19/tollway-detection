import cv2
import numpy as np

def draw_disappeared_vehicles_table(frame, disappeared_vehicles):
    """
    Draw an enhanced table showing disappeared vehicles by lane.
    
    Args:
        frame: The video frame to draw on
        disappeared_vehicles: Dictionary mapping lane numbers to lists of (vehicle_id, vehicle_type) tuples
        
    Returns:
        The frame with the table drawn on it
    """
    if not disappeared_vehicles:
        return frame
        
    # Set table parameters
    lanes = list(disappeared_vehicles.keys())
    table_rows = len(lanes) + 1  # Header + data rows
    row_height = 35
    table_height = table_rows * row_height + 20  # Add padding
    table_width = 400
    table_x = 20
    table_y = 20
    
    # Create transparent overlay for the table background
    overlay = frame.copy()
    cv2.rectangle(overlay, (table_x, table_y), 
                 (table_x + table_width, table_y + table_height), 
                 (40, 40, 40), -1)
    
    # Apply transparency
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw table border
    cv2.rectangle(frame, (table_x, table_y), 
                 (table_x + table_width, table_y + table_height), 
                 (200, 200, 200), 1)
    
    # Draw table header
    header_y = table_y + 10
    cv2.rectangle(frame, (table_x, table_y), 
                 (table_x + table_width, table_y + row_height), 
                 (60, 60, 100), -1)
    
    cv2.putText(frame, "Disappeared Vehicles by Lane", 
                (table_x + 10, header_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Column headers
    col1_x = table_x + 10  # Lane column
    col2_x = table_x + 100  # Vehicle info column
    
    # Draw column headers
    row_y = table_y + row_height
    cv2.line(frame, (table_x, row_y), (table_x + table_width, row_y), (200, 200, 200), 1)
    
    cv2.putText(frame, "Lane", (col1_x, row_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "Disappeared Vehicles (Latest 3)", (col2_x, row_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Vertical line between columns
    cv2.line(frame, (col2_x - 10, row_y), (col2_x - 10, table_y + table_height), 
             (200, 200, 200), 1)
    
    # Draw rows with alternating background for better readability
    row_y += row_height
    cv2.line(frame, (table_x, row_y), (table_x + table_width, row_y), (200, 200, 200), 1)
    
    # Add row data
    for i, lane in enumerate(sorted(lanes)):
        current_row_y = row_y + i * row_height
        
        # Draw alternating row background
        if i % 2 == 0:
            cv2.rectangle(frame, (table_x + 1, current_row_y + 1), 
                         (table_x + table_width - 1, current_row_y + row_height), 
                         (50, 50, 50), -1)
        
        # Draw lane number
        cv2.putText(frame, f"Lane {lane}", (col1_x, current_row_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw vehicle information with color coding by type
        vehicles = disappeared_vehicles[lane][-3:] if disappeared_vehicles[lane] else []  # Last 3 vehicles
        vehicle_text = ""
        
        for j, (vehicle_id, vehicle_type) in enumerate(vehicles):
            if j > 0:
                vehicle_text += ", "
            
            # Choose color based on vehicle type
            if "2-wheel" in vehicle_type:
                color = (120, 255, 120)  # Green for 2-wheel
            elif "4-wheel" in vehicle_type:
                color = (120, 120, 255)  # Blue for 4-wheel
            elif "6-wheel" in vehicle_type or "6-more-wheel" in vehicle_type:
                color = (120, 255, 255)  # Yellow for larger vehicles
            else:
                color = (200, 200, 200)  # Default gray
            
            # Draw individual vehicle with its own color
            type_text = vehicle_type.split("-")[0] + "-w"  # Shortened display (e.g., "2-w")
            veh_text = f"{type_text}(ID:{vehicle_id})"
            
            # Calculate position for this vehicle info
            if j == 0:
                text_x = col2_x
            else:
                # Measure previous text width to position the next one
                text_x = col2_x + len(vehicle_text) * 10
            
            cv2.putText(frame, veh_text, 
                      (text_x, current_row_y + 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            vehicle_text += veh_text + "  "
        
        # Draw horizontal line after each row
        cv2.line(frame, (table_x, current_row_y + row_height), 
                (table_x + table_width, current_row_y + row_height), 
                (200, 200, 200), 1)
    
    return frame

def draw_detection_boundary(frame, roi_start, width):
    """Draw a horizontal line showing the vehicle detection boundary."""
    cv2.line(frame, (0, roi_start), (width, roi_start), (0, 255, 0), 2)
    cv2.putText(frame, "Detection Zone", (width - 150, roi_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def create_vertical_table_image(disappeared_vehicles, width=800, height=600):
    """
    Create a separate image showing disappeared vehicles in a vertical table layout.
    Lanes are displayed as columns instead of rows.
    
    Args:
        disappeared_vehicles: Dictionary mapping lane numbers to lists of (vehicle_id, vehicle_type) tuples
        width: Width of the table image
        height: Height of the table image
        
    Returns:
        An image containing only the table
    """
    # Create a blank image for the table
    table_image = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark background
    
    # Always show 5 lanes (1-5), regardless of which lanes have disappeared vehicles
    lanes = [1, 2, 3, 4, 5]
    num_lanes = len(lanes)
    
    # Header dimensions
    header_height = 60
    
    # Column dimensions - now each column is for a lane (no Recent column)
    col_width = (width - 40) // num_lanes
    
    # Draw table title
    cv2.putText(table_image, "Disappeared Vehicles by Lane", 
               (width//2 - 200, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw the header row background
    cv2.rectangle(table_image, 
                 (20, header_height), 
                 (width - 20, header_height + 50), 
                 (60, 60, 100), -1)
                 
    # Draw lane headers - each column is a lane
    for i, lane in enumerate(lanes):
        # Calculate column position
        col_x = 20 + i * col_width
        
        # Draw vertical line for column divisions
        cv2.line(table_image, 
                (col_x, header_height), 
                (col_x, height - 20), 
                (150, 150, 150), 1)
        
        # Draw lane header
        cv2.putText(table_image, f"Lane {lane}", 
                   (col_x + col_width//2 - 30, header_height + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw right border
    cv2.line(table_image, 
            (width - 20, header_height), 
            (width - 20, height - 20), 
            (150, 150, 150), 1)
    
    # Draw horizontal grid lines
    for i in range(10):  # Show up to 10 latest disappeared vehicles
        row_y = header_height + 50 + i * 50
        cv2.line(table_image, 
                (20, row_y), 
                (width - 20, row_y), 
                (150, 150, 150), 1)
    
    # Draw the bottom border
    cv2.line(table_image, 
            (20, header_height + 50 + 10 * 50), 
            (width - 20, header_height + 50 + 10 * 50), 
            (150, 150, 150), 1)
    
    # Fill in the table with vehicle data
    for lane_idx, lane in enumerate(lanes):
        # Skip lanes that don't have any disappeared vehicles
        if lane not in disappeared_vehicles:
            continue
            
        vehicles = disappeared_vehicles[lane][-10:] if len(disappeared_vehicles[lane]) > 0 else []
        vehicles.reverse()  # Most recent at top
        
        col_x = 20 + lane_idx * col_width
        
        for row_idx, (vehicle_id, vehicle_type) in enumerate(vehicles):
            if row_idx >= 10:  # Only show the 10 most recent
                break
                
            row_y = header_height + 50 + row_idx * 50
            
            # Choose color based on vehicle type
            if "2-wheel" in vehicle_type:
                color = (120, 255, 120)  # Green for 2-wheel
            elif "4-wheel" in vehicle_type:
                color = (120, 120, 255)  # Blue for 4-wheel
            elif "6-wheel" in vehicle_type or "6-more-wheel" in vehicle_type:
                color = (120, 255, 255)  # Yellow for larger vehicles
            else:
                color = (200, 200, 200)  # Default gray
            
            # Use alternating background for better visibility
            if row_idx % 2 == 0:
                cv2.rectangle(table_image, 
                             (col_x + 1, row_y + 1), 
                             (col_x + col_width - 1, row_y + 50 - 1),
                             (50, 50, 50), -1)
            
            # Draw vehicle information
            type_text = vehicle_type.split("-")[0] + "-w"
            veh_text = f"{type_text} ID:{vehicle_id}"
            
            cv2.putText(table_image, veh_text, 
                       (col_x + 10, row_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    return table_image

def display_table_in_separate_window(disappeared_vehicles):
    """
    Create and display a separate window showing disappeared vehicles in a vertical table.
    
    Args:
        disappeared_vehicles: Dictionary mapping lane numbers to lists of (vehicle_id, vehicle_type) tuples
    """
    # Create table image
    table_image = create_vertical_table_image(disappeared_vehicles)
    
    # Display in a separate window
    cv2.imshow("Disappeared Vehicles Table", table_image)
