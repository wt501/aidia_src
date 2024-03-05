import os
import json
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# set memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


class AIConfig(object):
    def __init__(self, dataset_dir=None):
        """Common config class."""
        if dataset_dir is not None and not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"{dataset_dir} is not found.")
        self.dataset_dir = dataset_dir
        self.log_dir = None
        self.gpu_num = 0
        self.image_size = (0, 0)
        self.num_classes = 0
        self.total_batchsize = 0

        self.USE_MULTI_GPUS = False
        self.NAME = 'test'
        self.TASK = "Segmentation"
        self.DATASET_NUM = 1
        self.SEED = 12345

        self.SUBMODE = False
        self.DIR_SPLIT = True
        self.EARLY_STOPPING = False

        # training setting
        self.INPUT_SIZE = 224
        self.BATCH_SIZE = 32
        self.TRAIN_STEP = None
        self.VAL_STEP = None
        self.EPOCHS = 100
        self.LEARNING_RATE = 0.001
        self.SAVE_BEST = False
        self.LABELS = ["0"]
        self.N_SPLITS = 5

        self.MODEL = "YOLOv4"

        # self.DEPTH = 8
        # self.CROP_MODE = 'polygon'
        # self.SQUARE = False

        # image augmentatin
        self.EXPAND_X = 20
        self.EXPAND_Y = 20
        self.RANDOM_ROTATE = 30
        self.RANDOM_HFLIP = True
        self.RANDOM_VFLIP = True
        self.RANDOM_SHIFT = 0.2
        self.RANDOM_BRIGHTNESS = 0.1
        self.RANDOM_CONTRAST = 0.1
        self.RANDOM_SCALE = 0.2
        self.RANDOM_BLUR = 3.0  # 0 to n
        self.RANDOM_NOISE = 15
        self.RANDOM_SHEAR = 4

        self.build_params()
            
    def build_params(self):
        self.gpu_num = len(tf.config.list_logical_devices('GPU'))
        if self.USE_MULTI_GPUS and self.gpu_num > 1:
            self.total_batchsize = self.BATCH_SIZE * self.gpu_num
        else:
            self.total_batchsize = self.BATCH_SIZE
        if self.dataset_dir is not None:
            self.log_dir = os.path.join(self.dataset_dir, "data", self.NAME)
        self.image_size = (self.INPUT_SIZE, self.INPUT_SIZE)
        self.num_classes = len(self.LABELS)

    def load(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} is not found.")
        try:
            with open(json_path, encoding="utf-8") as f:
                dic = json.load(f)
                for key, value in dic.items():
                    if key == "dataset_dir":
                        continue
                    setattr(self, key, value)
        except Exception as e:
            try:    #  not UTF-8 json file handling
                with open(json_path) as f:
                    dic = json.load(f)
                    for key, value in dic.items():
                        if key == "dataset_dir":
                            continue
                        setattr(self, key, value)
            except Exception as e:
                raise ValueError(f"Failed to load config.json: {e}")

        self.build_params()

    def save(self, json_path):
        with open(json_path, mode='w', encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
