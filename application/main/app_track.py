# limit the number of cpus used by high performance libraries

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
lib_path = os.path.abspath(os.path.join('infrastructure', 'yolov5'))
sys.path.append(lib_path)

import torch
import torch.backends.cudnn as cudnn

import pandas as pd

from infrastructure.handlers.track import Tracker


if __name__== '__main__':
    tracker = Tracker(config_path="../../settings/config.yml")
    with torch.no_grad():
        tracker.detect()
    


