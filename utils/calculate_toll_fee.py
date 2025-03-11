def calculate_toll_fee(vehicle_type):
    if "4-wheel" in vehicle_type:
        return 30
    elif "6-wheel" in vehicle_type:
        return 75
    elif "6-more-wheel" in vehicle_type:
        return 120
    else:
        return 0