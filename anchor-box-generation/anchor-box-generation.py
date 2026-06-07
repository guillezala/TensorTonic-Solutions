def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size

    w_h = [[s*ar**0.5,s/ar**0.5] for s in scales for ar in aspect_ratios]

    boxes = []
        
    for i in range(feature_size):
        cy = (i+0.5)*stride
        for j in range(feature_size):
            cx = (j+0.5)*stride

            for w,h in w_h:
                boxes.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])

    return boxes