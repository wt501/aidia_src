import os
import tensorflow as tf
import numpy as np
import cv2
import subprocess
import glob
import random

from aidia.ai.dataset import Dataset
from aidia.ai.config import AIConfig
from aidia.ai import metrics

from aidia.image import det2merge

from aidia import LABEL_COLORMAP

from aidia.ai.models.yolov4.yolov4 import YOLO
from aidia.ai.models.yolov4.yolov4_generator import YOLODataGenerator
from aidia.ai.models.yolov4.yolov4_utils import image_preprocess


class DetectionModel(object):
    def __init__(self, config:AIConfig) -> None:
        self.config = config
        self.dataset = None
        self.model = None

        np.random.seed(self.config.SEED)
        tf.random.set_seed(self.config.SEED)
        random.seed(self.config.SEED)
    
    def set_config(self, config):
        self.config = config

    def build_dataset(self):
        self.dataset = Dataset(self.config)
    
    def load_dataset(self):
        self.dataset = Dataset(self.config, load=True)
    
    def build_model(self, mode):
        assert mode in ["train", "test"]

        if self.config.MODEL.find("YOLO") > -1:
            self.model = YOLO(self.config)
        else:
            raise NotImplementedError

        input_shape = (None, self.config.INPUT_SIZE, self.config.INPUT_SIZE, 3)
        self.model.build(input_shape=input_shape)
        self.model.compute_output_shape(input_shape=input_shape)

        optim = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        self.model.compile(optimizer=optim)
        if mode == "test":
            _wlist = os.path.join(self.config.log_dir, "weights", "*.h5")
            weights_path = sorted(glob.glob(_wlist))[-1]
            self.model.load_weights(weights_path)

    def train(self, custom_callbacks=None):
        checkpoint_path = os.path.join(self.config.log_dir, "weights")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, "{epoch:04d}.h5")

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=self.config.SAVE_BEST,
                save_weights_only=True,
                period=1 if self.config.SAVE_BEST else 10,
            ),
        ]
        if custom_callbacks:
            for c in custom_callbacks:
                callbacks.append(c)

        train_generator = YOLODataGenerator(self.dataset, self.config, mode="train")
        val_generator = YOLODataGenerator(self.dataset, self.config, mode="val")

        self.model.fit(
            train_generator,
            steps_per_epoch=self.dataset.train_steps,
            epochs=self.config.EPOCHS,
            verbose=0,
            validation_data=val_generator,
            validation_steps=self.dataset.val_steps,
            callbacks=callbacks
        )

    def stop_training(self):
        self.model.stop_training = True

    def evaluate(self, custom_callbacks=None):
        gt_count_per_class = np.array([0.0] * self.dataset.num_classes, dtype=float)
        tp_per_class = np.array([0.0] * self.dataset.num_classes, dtype=float)
        fp_per_class = np.array([0.0] * self.dataset.num_classes, dtype=float)
        sum_AP = 0.0
        ap_dictionary = {}
        for image_id in self.dataset.test_ids:
            # load image and annotation
            org_img = self.dataset.load_image(image_id, is_resize=False)
            anno_gt = self.dataset.get_yolo_bboxes(image_id)
            if len(anno_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = anno_gt[:, :4], anno_gt[:, 4]

            # ground truths
            bbox_dict_gt = []
            for i in range(len(bboxes_gt)):
                class_id = classes_gt[i]
                class_name = self.dataset.class_names[class_id]
                bbox = list(map(float, bboxes_gt[i]))
                bbox_dict_gt.append({"class_name": class_name,
                                     "bbox": bbox,
                                     "used": False})
                gt_count_per_class[class_id] += 1

            # prediction
            pred_bboxes = self.model.predict(org_img)
            
            bbox_dict_pred = []
            for bbox_pred in pred_bboxes:
                # xmin, ymin, xmax, ymax = list(map(str, map(int, bbox[:4])))
                bbox = list(map(float, bbox_pred[:4]))
                score = bbox_pred[4]
                class_id = int(bbox_pred[5])
                class_name = self.dataset.class_names[class_id]
                score = '%.4f' % score
                bbox_dict_pred.append({"class_id": class_id,
                                       "class_name": class_name,
                                       "confidence": score,
                                       "bbox": bbox})
            bbox_dict_pred.sort(key=lambda x:float(x['confidence']), reverse=True)

            merge = det2merge(org_img, bbox_dict_pred)
            cv2.imwrite(f"debug_data/test{image_id}.png", merge)


            # count True Positive and False Positive
            for pred in bbox_dict_pred:
                overlap_max = -1
                gt_match = -1

                cls_id = pred["class_id"]
                cls_pred = pred["class_name"]
                bb_pred = pred["bbox"]

                for gt in bbox_dict_gt:
                    cls_gt = gt["class_name"]
                    if cls_gt != cls_pred:
                        continue
                    bb_gt = gt["bbox"]
                    bi = [max(bb_pred[0], bb_gt[0]),
                          max(bb_pred[1], bb_gt[1]),
                          min(bb_pred[2], bb_gt[2]),
                          min(bb_pred[3], bb_gt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        area_pred = (bb_pred[2] - bb_pred[0] + 1) * (bb_pred[3] - bb_pred[1] + 1)
                        area_gt = (bb_gt[2] - bb_gt[0] + 1) * (bb_gt[3] - bb_gt[1] + 1)
                        ua = area_pred + area_gt - iw * ih
                        ov = (iw * ih) / ua
                        if ov > overlap_max:
                            overlap_max = ov
                            gt_match = gt
            
                # set minimum overlap (AP50)
                iou_threshold = 0.5  # TODO:user configuration param
                if overlap_max >= iou_threshold:
                    if not bool(gt_match["used"]):
                        gt_match["used"] = True
                        tp_per_class[cls_id] += 1.0
                    else:
                        fp_per_class[cls_id] += 1.0
                else:
                    fp_per_class[cls_id] += 1.0

        # compute precision/recall
        total_gt = np.sum(gt_count_per_class)
        total_tp = np.sum(tp_per_class)
        total_fp = np.sum(fp_per_class)

        precision_per_class = tp_per_class / (fp_per_class + tp_per_class)
        recall_per_class = tp_per_class / gt_count_per_class

        precision = total_tp / (total_fp + total_tp)
        recall = total_tp / total_gt

        result = {
            "precision": precision,
            "recall": recall
        }
        return result
    
    def predict_by_id(self, image_id, thresh=0.5):
        src_img = self.dataset.load_image(image_id)
        gt_mask_data = self.dataset.load_masks(image_id)
        img = np.array(src_img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        pred = self.model.predict(img, batch_size=1, verbose=0)[0]
        merge = src_img.astype(float)
        gt_merge = src_img.astype(float)
        for c in range(pred.shape[2] - 1):
            m = pred[:, :, c + 1]
            gt_m = gt_mask_data[:, :, c + 1]
            color = LABEL_COLORMAP[c + 2]
            color = color[::-1]
            indexes = np.where(m >= thresh)
            mask = np.zeros_like(src_img)
            gt_mask = np.zeros_like(src_img)
            for x in range(3):
                binary = np.zeros_like(m, dtype=float)
                binary[indexes] = float(color[x])
                mask[..., x] = binary
                gt_mask[..., x] = gt_m * float(color[x])
            merge += mask.astype(float)
            gt_merge += gt_mask.astype(float)

        fontsize = 2
        fontcolor = (255, 255, 255) if np.mean(src_img[:, :100, :]) < 128 else (0, 0, 0)  # select color depending on background
        fontweight = 2

        cv2.putText(src_img, "original", (0, 50), cv2.FONT_HERSHEY_PLAIN, fontsize, fontcolor, fontweight, cv2.LINE_AA)

        merge[merge > 255] = 255.0
        merge = merge.astype(np.uint8)
        cv2.putText(merge, "AI", (0, 50), cv2.FONT_HERSHEY_PLAIN, fontsize, fontcolor, fontweight, cv2.LINE_AA)

        gt_merge[gt_merge > 255] = 255.0
        gt_merge = gt_merge.astype(np.uint8)
        cv2.putText(gt_merge, "human", (0, 50), cv2.FONT_HERSHEY_PLAIN, fontsize, fontcolor, fontweight, cv2.LINE_AA)

        concat = np.concatenate([src_img, merge, gt_merge], axis=1)
        label_box = np.zeros((50 ,concat.shape[1], 3), np.uint8)
        cv2.rectangle(label_box, (0, 0), (label_box.shape[1], label_box.shape[0]), color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_4, shift=0)
        for c in range(pred.shape[2] - 1):
            color = LABEL_COLORMAP[c + 2][::-1]
            label = self.dataset.class_names[c]
            cv2.putText(label_box, "*", (100*c + 50*c, 40), cv2.FONT_HERSHEY_PLAIN, fontsize, color, fontweight, cv2.LINE_AA)
            cv2.putText(label_box, label, (20 + 100*c + 50*c, 40), cv2.FONT_HERSHEY_PLAIN, fontsize, color, fontweight, cv2.LINE_AA)
        concat = np.concatenate([concat, label_box], axis=0)

        return concat
    
    def convert2onnx(self):
        saved_model_path = os.path.join(self.config.log_dir, 'saved_model')
        onnx_path = os.path.join(self.config.log_dir, "model.onnx")
        if os.path.exists(onnx_path):
            return
        self.model.save(saved_model_path)
        subprocess.run(['python', '-m', 'tf2onnx.convert',
                        '--saved-model', saved_model_path,
                        '--output', onnx_path,
                        '--opset', '11'])

