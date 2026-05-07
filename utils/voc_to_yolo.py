import xml.etree.ElementTree as ET
import os
from pathlib import Path
import logging

def convert_bbox(size, box):
    """
    Convert Pascal VOC bbox to YOLO format.
    size: (width, height)
    box: (xmin, xmax, ymin, ymax)
    Returns: (x_center, y_center, width, height) normalized to [0, 1]
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def voc_to_yolo(xml_path, output_path, class_map):
    """
    Parses a single XML file and writes a YOLO .txt file.
    Only keeps classes defined in class_map.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size_elem = root.find('size')
        if size_elem is None:
            logging.warning(f"No size found in {xml_path}. Skipping.")
            return False
            
        w = int(size_elem.find('width').text)
        h = int(size_elem.find('height').text)
        
        if w == 0 or h == 0:
            logging.warning(f"Invalid image size in {xml_path}. Skipping.")
            return False

        yolo_lines = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_map:
                continue
                
            cls_id = class_map[cls_name]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            
            # Clip values to image boundaries
            b = (max(0, b[0]), min(w, b[1]), max(0, b[2]), min(h, b[3]))
            
            bb = convert_bbox((w, h), b)
            yolo_lines.append(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}")

        with open(output_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        return True
    except Exception as e:
        logging.error(f"Error processing {xml_path}: {e}")
        return False
