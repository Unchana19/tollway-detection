def get_center_of_bbox(bbox):
    x, _, w, _ = bbox
    return (x + w / 2, 0)