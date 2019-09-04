import numpy as np
import cv2
import io
import time
from edgetpu.detection.engine import DetectionEngine

# Function to read labels from text files.
def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret

class ObjectDetector:
    
    def init(self, model_file="mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite",
             label_file = "coco_labels.txt"):
        self.model_file = model_file
        self.label_file = label_file
        self.labels =  ReadLabelFile(self.label_file)
        self.engine = DetectionEngine(self.model_file)
    
    def detect(self, input_frame):
        if (self.labels == '' or self.engine == ''):
            print("Detector is not initialized!")
            return []
        objects = self.engine.DetectWithInputTensor(input_frame.flatten(), threshold=0.5, top_k=10)
        _, width, height, channels = self.engine.get_input_tensor_shape()
        detected_objects = []
        if objects:
            for obj in objects:
                box = obj.bounding_box.flatten().tolist()
                box_left = int(box[0]*width)
                box_top = int(box[1]*height)
                box_right = int(box[2]*width)
                box_bottom = int(box[3]*height)
                percentage = int(obj.score * 100)
                label = self.labels[obj.label_id]
                object_info = {'box_left': box_left, 'box_right': box_right, 
                               'box_top': box_top, 'box_bottom': box_bottom,
                               'percentage': percentage, 'label': label}
                detected_objects.append(object_info)
                
        return detected_objects
    
