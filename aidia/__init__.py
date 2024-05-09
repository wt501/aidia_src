# coding: utf-8
import os
import logging

__appname__ = "Aidia"
__version__ = "1.2.4.1"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.expanduser("~")

app_cfg_dir = os.path.join(HOME_DIR, ".aidia")
if not os.path.exists(app_cfg_dir):
    os.mkdir(app_cfg_dir)
pretrained_dir = os.path.join(app_cfg_dir, "pretrained")
if not os.path.exists(pretrained_dir):
    os.mkdir(pretrained_dir)

aidia_logger = logging.getLogger("Aidia")
aidia_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s:%(filename)s:%(lineno)d - %(message)s')
file_handler = logging.FileHandler(os.path.join(app_cfg_dir, "errors.log"))
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
aidia_logger.addHandler(file_handler)
aidia_logger.addHandler(stream_handler)

# task name definition
CLS = "Classification"
DET = "Detection"
SEG = "Segmentation"
MNIST = "MNIST Test"

# model definition
CLS_MODEL = ["EfficientNetv2-s"]
DET_MODEL = ["YOLOv4", "YOLOv4-tiny", "YOLOv3", "YOLOv3-tiny"]
SEG_MODEL = ["U-Net"]

CFONT = "Arial"
CFONT_SIZE = 10

if os.name == "posix":
    CFONT = "Hiragino Sans"
    CFONT_SIZE = 12
elif os.name == "nt":
    # CFONT = "Meiryo"
    CFONT = "BIZ UDゴシック"
    # CFONT = "MS Gothic"
    # CFONT = "Yu Gothic"
    # CFONT = "Yu Gothic UI"
    # CFONT = "Yu Mincho"
    CFONT_SIZE = 10

del os

# generated by imgviz.label_colormap(200)
LABEL_COLORMAP = [[0, 0, 0], [200, 0, 0], [0, 200, 0], [0, 0, 200], [198, 0, 200], [0, 200, 198], [198, 200, 0], [132, 200, 0], [99, 0, 200], [200, 0, 136], [100, 200, 199], [200, 133, 133], [0, 200, 0], [200, 99, 0], [0, 200, 0], [136, 200, 0], [0, 104, 200], [199, 100, 200], [0, 200, 132], [133, 200, 133], [200, 198, 0], [200, 66, 0], [71, 200, 0], [200, 198, 0], [100, 100, 200], [200, 67, 136], [67, 200, 133], [200, 199, 133], [0, 0, 200], [200, 0, 104], [0, 200, 99], [200, 199, 100], [0, 0, 200], [132, 0, 200], [0, 136, 200], [133, 133, 200], [198, 0, 200], [200, 0, 71], [100, 200, 100], [200, 133, 67], [66, 0, 200], [198, 0, 200], [67, 136, 200], [199, 133, 200], [0, 200, 198], [200, 100, 100], [0, 200, 66], [136, 200, 67], [0, 71, 200], [133, 67, 200], [0, 200, 198], [133, 200, 199], [200, 200, 200], [200, 67, 67], [67, 200, 67], [200, 198, 67], [67, 67, 200], [198, 67, 200], [67, 200, 198], [200, 0, 0], [200, 0, 0], [52, 200, 0], [200, 160, 0], [47, 0, 200], [200, 0, 160], [50, 200, 198], [200, 160, 160], [200, 0, 0], [200, 0, 0], [151, 200, 0], [200, 113, 0], [146, 0, 200], [200, 0, 118], [151, 200, 199], [200, 115, 115], [104, 200, 0], [200, 75, 0], [38, 200, 0], [169, 200, 0], [50, 103, 200], [200, 80, 162], [34, 200, 132], [168, 200, 133], [200, 132, 0], [200, 56, 0], [104, 200, 0], [200, 169, 0], [150, 100, 200], [200, 57, 114], [100, 200, 133], [200, 171, 115], [99, 0, 200], [200, 0, 80], [50, 200, 100], [200, 159, 80], [33, 0, 200], [165, 0, 200], [34, 135, 200], [166, 133, 200], [200, 0, 136], [200, 0, 61], [152, 200, 100], [200, 111, 57], [99, 0, 200], [200, 0, 174], [100, 136, 200], [200, 115, 173], [100, 200, 199], [200, 80, 80], [34, 200, 65], [167, 200, 67], [34, 69, 200], [164, 67, 200], [34, 200, 198], [167, 200, 200], [200, 133, 133], [200, 57, 57], [101, 200, 67], [200, 171, 57], [98, 67, 200], [200, 57, 171], [100, 200, 199], [200, 172, 172], [0, 200, 0], [200, 47, 0], [0, 200, 0], [160, 200, 0], [0, 52, 200], [198, 50, 200], [0, 200, 160], [160, 200, 160], [200, 99, 0], [200, 33, 0], [80, 200, 0], [200, 165, 0], [100, 50, 200], [200, 34, 135], [80, 200, 159], [200, 166, 133], [200, 146, 0], [118, 200, 0], [0, 151, 200], [199, 151, 200], [0, 200, 113], [115, 200, 115], [136, 200, 0], [200, 99, 0], [61, 200, 0], [174, 200, 0], [100, 152, 200], [200, 100, 136], [57, 200, 114], [173, 200, 115], [0, 104, 200], [200, 50, 103], [0, 200, 80], [162, 200, 80], [0, 38, 200], [132, 34, 200], [0, 169, 200], [133, 168, 200], [199, 100, 200], [200, 34, 69], [80, 200, 80], [200, 164, 67], [65, 34, 200], [198, 34, 200], [67, 167, 200], [200, 167, 200], [0, 200, 132], [200, 150, 100], [0, 200, 56], [114, 200, 57], [0, 104, 200], [133, 100, 200], [0, 200, 169], [115, 200, 171], [133, 200, 133], [200, 98, 67], [57, 200, 57], [171, 200, 57], [67, 101, 200], [199, 100, 200], [57, 200, 171], [172, 200, 172], [200, 198, 0], [200, 38, 0], [42, 200, 0], [200, 198, 0], [50, 50, 200], [200, 40, 160], [40, 200, 157], [200, 200, 160], [200, 66, 0], [200, 28, 0], [122, 200, 0], [200, 141, 0], [149, 50, 200], [200, 29, 118], [120, 200, 160], [200, 143, 115], [71, 200, 0], [200, 118, 0], [33, 200, 0], [146, 200, 0], [50, 152, 200], [200, 120, 161], [29, 200, 114], [145, 200, 115], [200, 198, 0], [200, 85, 0], [89, 200, 0], [200, 198, 0], [151, 151, 200], [200, 86, 116], [86, 200, 113], [200, 199, 115], [100, 100, 200], [200, 40, 81], [40, 200, 78], [200, 199, 80], [34, 34, 200], [167, 34, 200], [34, 167, 200], [167, 167, 200], [200, 67, 136], [200, 29, 61], [122, 200, 80], [200, 141, 57], [100, 34, 200], [200, 29, 174], [100, 168, 200], [200, 144, 173], [67, 200, 133], [200, 120, 80], [29, 200, 57], [145, 200, 57], [34, 104, 200], [166, 100, 200], [29, 200, 170], [144, 200, 171], [200, 199, 133], [200, 84, 57], [87, 200, 57], [200, 198, 57], [100, 100, 200], [200, 86, 172], [86, 200, 169], [200, 200, 172]]