class OPT:
    def __init__(self, config:dict) -> None:
        
        
        self.output = config['output']
        self.source = config['source']
        self.yolo_weights = config['yolo_weights']
        self.deep_sort_weights = config['deep_sort_weights']
        self.show_vid = config['show_vid']
        self.save_vid = config['save_vid']
        self.save_txt = config['save_txt']
        self.save_csv = config['save_csv']
        self.imgsz = config['imgsz']
        self.evaluate = config['evaluate']
        self.half = config['half']
        self.config_deepsort = config['config_deepsort']
        self.visualize = config['visualize']
        self.fourcc = config['fourcc']
        self.device = config['device']
        self.augment = config['augment']
        self.dnn = config['dnn']
        self.conf_thres = config['conf_thres']
        self.iou_thres = config['iou_thres']
        self.classes = config['classes']
        self.agnostic_nms = config['agnostic_nms']
        self.max_det = config['max_det']
        self.upload_db = config['upload_db']
        self.upper_ratio = config['upper_ratio']
        self.lower_ratio = config['lower_ratio']



