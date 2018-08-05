import sys
import os
from datetime import datetime
# 加载预训练检测模型 : ssd
PATH_TO_DETECTION_PB = './data/twostepclassify/detection_models/ssd_mobilenet_v2_coco_2018_03_29.pb'   # 图协议文件 
PATH_TO_DETECTION_LABELS = './data/twostepclassify/detection_models/mscoco_label_map.pbtxt'  # 标签
DETECTION_NUM_CLASSES = 90

# 加载预训练分类模型 : inception
PATH_TO_CLASSIFY_PB = './data/twostepclassify/classify_models/vehicle_inception_v4_freeze.pb'   # 图协议文件
PATH_TO_CLASSIFY_LABELS = './data/labels.txt'  # 标签
CLASSIFY_NUM_CLASSES = 764

# 'twostep' ,'onestep' or 'cam'
DETECTOR_MODE = 'twostep'

sys.path.append("D:/Users/Documents/AI/ThirdParty/models/research")

def get_models_config():
    DETECT_CONFIG ={'pb':PATH_TO_DETECTION_PB,'labels':PATH_TO_DETECTION_LABELS,
        'num_class':DETECTION_NUM_CLASSES}
    CLASSIFY_CONFIG = {'pb':PATH_TO_CLASSIFY_PB,'labels':PATH_TO_CLASSIFY_LABELS,
        'num_class':CLASSIFY_NUM_CLASSES}
    return DETECT_CONFIG, CLASSIFY_CONFIG

def get_log_dir():
    # path to save logs
    if not os.path.exists('./log'):
        os.mkdir('./log')
    log_name = datetime.now().strftime("log-%Y%m%d-%H%M%S.txt")
    LOG_PATH = './log/' + log_name
    return LOG_PATH


def get_detector_mode():
    return DETECTOR_MODE