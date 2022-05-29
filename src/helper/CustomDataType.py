import yaml
from datetime import datetime
import sys
from utils.common import read_yml

def read_yml(path: str):
    with open(path, 'r') as file:
        text = yaml.safe_load(file)
    return text


class OPTConfig:
    def __init__(self, config_file_path):
        config_file = read_yml(config_file_path)
        self.yolo_model = config_file['yolo_model']
        self.deep_sort_model = config_file['deep_sort_model']
        self.source = config_file['source']
        self.output = config_file['output']
        self.imgsz = config_file['imgsz']
        self.conf_thres = config_file['conf_thres']
        self.iou_thres = config_file['iou_thres']
        self.fourcc = config_file['fourcc']
        self.device = config_file['device']
        self.show_vid = config_file['show_vid']
        self.save_vid = config_file['save_vid']
        self.save_txt = config_file['save_txt']

        # class 0 is person, 1 2 3 5 7
        self.classes = config_file['classes']
        self.agnostic_nms = config_file['agnostic_nms']
        self.augment = config_file['augment']
        self.update = config_file['update']
        self.evaluate = config_file['evaluate']
        self.config_deepsort = config_file['config_deepsort']
        self.half = config_file['half']
        self.visualize = config_file['visualize']
        self.max_det = config_file['max_det']
        self.save_crop = config_file['save_crop']
        self.dnn = config_file['dnn']
        self.project = config_file['project']
        self.name = config_file['name']
        self.exist_ok = config_file['exist_ok']

        # self.imgsz *= 2 if len(self.imgsz) == 1 else 1  # expand
        # self.project = ROOT_path / 'runs/track'


class VehicleInfo:
    def __init__(self):
        self.in_time = datetime.max
        self.exit_time = datetime.max
        self.type = ''
        self.lane = [0, 0]
        self.temporarily_disappear = 0


class FrameInfo:
    def __init__(self):
        self.first = True
        self.time = datetime.max
        self.nth_frame = 0
        self.n_vehicles_at_time = 0
        self.IDs_vehicles = []
