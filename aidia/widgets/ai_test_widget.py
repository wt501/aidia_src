import os
import cv2
import numpy as np

from onnxruntime import InferenceSession
from qtpy import QtWidgets

from aidia import CLS, DET, SEG
from aidia.image import convert_dtype, mask2polygon
from aidia.ai.config import AIConfig

class AITestWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
     
    def generate_shapes(self, img, log_dir, epsilon=None):
        h, w = img.shape[:2]
        if img.dtype == np.uint16:
            img = convert_dtype(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        config = AIConfig()
        config.load(os.path.join(log_dir, "config.json"))

        onnx_path = os.path.join(log_dir, "model.onnx")
        model = InferenceSession(onnx_path)

        if config.TASK == SEG:
            img = cv2.resize(img, config.image_size)
            img = img.astype(np.float32)
            img = img / 255.0
            inputs = np.expand_dims(img, axis=0)
            input_name = model.get_inputs()[0].name
            result = model.run([], {input_name: inputs})[0]
            masks = np.where(result[0] >= 0.5, 255, 0)
            masks = masks.astype(np.uint8)
            masks = cv2.resize(masks, (w, h), cv2.INTER_NEAREST)
            shapes = mask2polygon(masks, config.LABELS, epsilon)
            return shapes
        else:
            raise NotImplementedError
