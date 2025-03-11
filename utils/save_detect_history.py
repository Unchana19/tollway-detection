import csv
import os

def save_detection_history_to_csv(detection_history, filename="vehicle_detection_data.csv"):
  # Create outputs directory if it doesn't exist
  output_dir = "outputs"
  os.makedirs(output_dir, exist_ok=True)
  
  # Count vehicles by class
  vehicle_counts = {}
  # Count vehicles by lane
  lane_counts = {}
  # Count vehicles by lane and class
  lane_vehicle_counts = {}
  
  for record in detection_history:
    v_type = record['vehicle_type']
    lane = record['lane']
    
    # Count by vehicle type
    if v_type in vehicle_counts:
      vehicle_counts[v_type] += 1
    else:
      vehicle_counts[v_type] = 1
    
    # Count by lane
    if lane in lane_counts:
      lane_counts[lane] += 1
    else:
      lane_counts[lane] = 1
    
    # Count by lane and vehicle type
    if lane not in lane_vehicle_counts:
      lane_vehicle_counts[lane] = {}
    
    if v_type in lane_vehicle_counts[lane]:
      lane_vehicle_counts[lane][v_type] += 1
    else:
      lane_vehicle_counts[lane][v_type] = 1
  
  # Save full detection history
  filepath = os.path.join(output_dir, filename)
  with open(filepath, 'w', newline='') as csvfile:
    fieldnames = ['id', 'vehicle_type', 'lane', 'time', 'payment_status', 'toll_fee']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for record in detection_history:
      writer.writerow(record)
  
  # Save lane-specific vehicle counts
  lane_counts_filepath = os.path.join(output_dir, 'lane_vehicle_counts.csv')
  with open(lane_counts_filepath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Lane', 'Vehicle Type', 'Count'])
    
    # Sort by lane number for better readability
    for lane in sorted(lane_vehicle_counts.keys()):
      for v_type, count in lane_vehicle_counts[lane].items():
        writer.writerow([lane, v_type, count])
      # Add a total row for each lane
      writer.writerow([lane, 'TOTAL', lane_counts[lane]])
      # Add an empty row for better readability
      writer.writerow(['', '', ''])
  
  print(f"Detection history saved to {filepath}")
  print(f"Lane-specific vehicle counts saved to {lane_counts_filepath}")