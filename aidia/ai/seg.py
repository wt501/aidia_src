import os
import tensorflow as tf
import numpy as np
import subprocess
import glob
import random
import imgaug
import imgaug.augmenters as iaa
from sklearn import metrics

from aidia.ai.dataset import Dataset
from aidia.ai.config import AIConfig
from aidia.ai.models.unet import UNet
# from aidia.ai import metrics

from aidia import image
from aidia import utils

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")
plt.rcParams["font.size"] = 15

np.set_printoptions(suppress=True)


def mask_iou(pred, gt):
    pred_list = image.mask2rect(pred)
    gt_list = image.mask2rect(gt)
    
    ious = []
    for pred_rect in pred_list:
        best_iou = 0.0
        for gt_rect in gt_list:
           iou = calc_iou(pred_rect, gt_rect)
           if iou > best_iou:
               best_iou = iou
        ious.append(best_iou)

    return ious

def calc_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max((inter_x2 - inter_x1), 0) * max((inter_y2 - inter_y1), 0)
    area_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area_1 + area_2 - inter_area

    iou = inter_area / union_area
    return iou

def eval_on_iou(y_true, y_pred):
    tp = 0
    fp = 0
    num_gt = 0
    for i in range(y_true.shape[0]):
        pred_mask = y_pred[i]
        gt_mask = y_true[i]
        iou_list = mask_iou(pred_mask, gt_mask)
        for iou in iou_list:
            if iou >= 0.5:
                tp += 1
            else:
                fp += 1
        num_gt += len(image.mask2rect(gt_mask))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (num_gt + 1e-12)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    return [precision, recall, f1]

def common_metrics(tp, tn, fp, fn):
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    return [acc, precision, recall, specificity, f1]


class SegmentationModel(object):
    def __init__(self, config:AIConfig) -> None:
        self.config = config
        self.dataset = None
        self.model = None

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        tf.random.set_seed(self.config.SEED)
        imgaug.seed(self.config.SEED)

    def set_config(self, config):
        self.config = config

    def build_dataset(self):
        self.dataset = Dataset(self.config)
    
    def load_dataset(self):
        self.dataset = Dataset(self.config, load=True)
    
    def build_model(self, mode):
        assert mode in ["train", "test"]
        self.model = UNet(self.config.num_classes)

        input_shape = (None, self.config.INPUT_SIZE, self.config.INPUT_SIZE, 3)
        self.model.build(input_shape=input_shape)
        self.model.compute_output_shape(input_shape=input_shape)

        # if mode == "test":
        #     custom_metrics.append(metrics.MultiMetrics())
        #     for thresh in THRESH_LIST:
        #         custom_metrics.append(metrics.MultiMetrics(thresh, name=f"MM_{thresh}"))
        #     for class_id in range(self.config.num_classes + 1):
        #         custom_metrics.append(metrics.MultiMetrics(class_id=class_id, name=f"MM_{class_id}"))
        #         for thresh in THRESH_LIST:
        #             custom_metrics.append(metrics.MultiMetrics(thresh, class_id, name=f"MM_{class_id}_{thresh}"))

        optim = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        self.model.compile(
            optimizer=optim,
            loss=tf.keras.losses.BinaryCrossentropy())
        
        if mode == "test":
             # select latest weights
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

        train_generator = SegDataGenerator(self.dataset, self.config, mode="train")
        val_generator = SegDataGenerator(self.dataset, self.config, mode="val")

        train_generator = tf.data.Dataset.from_generator(
            train_generator.flow, (tf.float32, tf.float32),
            output_shapes=(self.model.input_shape, self.model.output_shape)
        )
        val_generator = tf.data.Dataset.from_generator(
            val_generator.flow, (tf.float32, tf.float32),
            output_shapes=(self.model.input_shape, self.model.output_shape)
        )
        
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

    def evaluate(self, cb_widget=None):
        res = {}
        y_true = []
        y_pred = []
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # predict all test data
        for i, image_id in enumerate(self.dataset.test_ids):
            if cb_widget is not None:
                cb_widget.notifyMessage.emit(f"Predicting... {i+1} / {self.dataset.num_test}")
                cb_widget.progressValue.emit(int((i+1) / self.dataset.num_test * 100))
            img = self.dataset.load_image(image_id)
            mask = self.dataset.load_masks(image_id)
            inputs = image.preprocessing(img, is_tensor=True)
            p = self.model.predict(inputs, verbose=0)[0]
            mask = mask[..., 1:] # exclude background
            p = p[..., 1:]
            y_true.append(mask)
            y_pred.append(p)
            if i == 20:  # TODO
                break
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if cb_widget is not None:
            cb_widget.notifyMessage.emit("Calculating...")

        # evaluation
        THRESH = 0.5
        eval_dict = {}
        eval_dict["Metrics"] = [
            "Accuracy", "Precision", "Recall", "Specificity",
            "F1", "ROC Curve AUC", "Average Precision",
            "Precision (Detection)", "Recall (Detection)", "F1 (Detection)"
        ]
        if self.config.num_classes > 1:
            num_results = self.config.num_classes
            sum_acc = 0.0
            sum_pre = 0.0
            sum_rec = 0.0
            sum_spe = 0.0
            sum_f1 = 0.0
            sum_auc = 0.0
            sum_ap = 0.0
            sum_pre_det = 0.0
            sum_rec_det = 0.0
            sum_f1_det = 0.0
            if cb_widget is not None:
                cb_widget.progressValue.emit(0)
            for class_id in range(self.config.num_classes):
                if cb_widget is not None:
                    cb_widget.notifyMessage.emit(f"{class_id + 1} / {self.config.num_classes}")
                    cb_widget.progressValue.emit(int((class_id + 1) / self.config.num_classes * 100))

                # prepare class result directories
                class_name = self.config.LABELS[class_id]
                class_dir = os.path.join(self.config.log_dir, class_name)
                if not os.path.exists(class_dir):
                    os.mkdir(class_dir)

                # get result by class
                yt = y_true[..., class_id]
                yp = y_pred[..., class_id]

                if np.max(yt) == 0 or np.max(yp) == 0:  # no ground truth data
                    num_results -= 1
                    continue

                # ROC curve and PR curve
                yt_flat = yt.ravel()
                yp_flat_prob = yp.ravel()

                fpr, tpr, thresholds = metrics.roc_curve(yt_flat, yp_flat_prob)
                ax.plot(fpr, tpr)
                ax.set_title(f"ROC Curve ({class_name})")
                ax.set_xlabel('FPR')
                ax.set_ylabel('TPR')
                ax.grid()
                fig.savefig(os.path.join(class_dir, "roc.png"))
                ax.clear()

                pres, recs, thresholds = metrics.precision_recall_curve(yt_flat, yp_flat_prob)
                ax.plot(pres, recs)
                ax.set_title(f"PR Curve ({class_name})")
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.grid()
                fig.savefig(os.path.join(class_dir, "pr.png"))
                ax.clear()

                # AUC and AP
                auc = metrics.roc_auc_score(yt_flat, yp_flat_prob)
                ap = metrics.average_precision_score(yt_flat, yp_flat_prob)
                sum_auc += auc
                sum_ap += ap

                # common metrics
                yp[yp >= THRESH] = 1
                yp[yp < THRESH] = 0

                pre_det, rec_det, f1_det = eval_on_iou(yt, yp)
                sum_pre_det += pre_det
                sum_rec_det += rec_det
                sum_f1_det += f1_det

                yt_flat = yt.ravel()
                yp_flat = yp.ravel()

                cm = metrics.confusion_matrix(yt_flat, yp_flat)
                tn, fp, fn, tp = cm.ravel()
                acc, pre, rec, spe, f1 = common_metrics(tp, tn, fp, fn)
                sum_acc += acc
                sum_pre += pre
                sum_rec += rec
                sum_spe += spe
                sum_f1 += f1

                # add result by class
                eval_dict[class_name] = [
                    acc, pre, rec, spe, f1, auc, ap, pre_det, rec_det, f1_det
                ]
            
            # macro mean
            acc = sum_acc / num_results
            pre = sum_pre / num_results
            rec = sum_rec / num_results
            spe = sum_spe / num_results
            f1 = sum_f1 / num_results
            auc = sum_auc / num_results
            ap = sum_ap / num_results
            pre_det = sum_pre_det / num_results
            rec_det = sum_rec_det / num_results
            f1_det = sum_f1_det / num_results

            # confusion matrix
            yt = np.argmax(y_true, axis=-1)
            yp = np.argmax(y_pred, axis=-1)
            yt = yt.ravel()
            yp = yp.ravel()
            cm = metrics.confusion_matrix(yt, yp)
            cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                    display_labels=self.config.LABELS)
            cm_disp.plot(ax=ax)
            filename = os.path.join(self.config.log_dir, "confusion_matrix.png")
            fig.savefig(filename)
            img = image.fig2img(fig)

            # save eval dict
            eval_dict["(Macro Mean)"] = [
                acc, pre, rec, spe, f1, auc, ap, pre_det, rec_det, f1_det
            ]
            utils.save_dict_to_excel(eval_dict, os.path.join(self.config.log_dir, "eval.xlsx"))
           
           
        else:  # binary classification
            y_true = y_true[..., 0]
            y_pred = y_pred[..., 0]
            class_name = self.config.LABELS[0]

            # AUC and AP
            yt = y_true.ravel()
            yp = y_pred.ravel()

            fpr, tpr, thresholds = metrics.roc_curve(yt_flat, yp_flat_prob)
            ax.plot(fpr, tpr)
            ax.set_title(f"ROC Curve ({class_name})")
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.grid()
            fig.savefig(os.path.join(self.config.log_dir, "roc.png"))
            ax.clear()

            pres, recs, thresholds = metrics.precision_recall_curve(yt_flat, yp_flat_prob)
            ax.plot(pres, recs)
            ax.set_title(f"PR Curve ({class_name})")
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.grid()
            fig.savefig(os.path.join(self.config.log_dir, "pr.png"))
            ax.clear()
        
            auc = metrics.roc_auc_score(yt, yp)
            ap = metrics.average_precision_score(yt, yp)

            y_pred[y_pred >= THRESH] = 1
            y_pred[y_pred < THRESH] = 0

            pre_det, rec_det, f1_det = eval_on_iou(y_true, y_pred)

            y_true = y_true.ravel()
            y_pred = y_pred.ravel()

            cm = metrics.confusion_matrix(y_true, y_pred)

            # confusion matrix
            cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
            cm_disp.plot(ax=ax)
            filename = os.path.join(self.config.log_dir, "confusion_matrix.png")
            fig.savefig(filename)
            img = image.fig2img(fig)

            tn, fp, fn, tp = cm.ravel()
            acc, pre, rec, spe, f1 = common_metrics(tp, tn, fp, fn)

            eval_dict[class_name] = [
                acc, pre, rec, spe, f1, auc, ap, pre_det, rec_det, f1_det
            ]

        res = {
            "Accuracy": acc,
            "Precision": pre,
            "Recall": rec,
            "Specificity": spe,
            "F1": f1,
            "ROC Curve AUC": auc,
            "Average Precision": ap,
            "Precision (Detection)": pre_det,
            "Recall (Detection)": rec_det,
            "F1 (Detection)": f1_det,
            "img": img,
        }
        return res

    def predict_by_id(self, image_id, thresh=0.5):
        src_img = self.dataset.load_image(image_id)
        gt_mask_data = self.dataset.load_masks(image_id)
        img = np.array(src_img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        pred = self.model.predict(img, batch_size=1, verbose=0)[0]
        concat = image.mask2merge(src_img, pred, self.dataset.class_names, gt_mask_data, thresh)
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


class SegDataGenerator(object):
    def __init__(self, dataset:Dataset, config:AIConfig, mode="train") -> None:
        assert mode in ["train", "val", "test"]

        self.dataset = dataset
        self.config = config
        self.mode = mode

        self.augseq = None
        self.images = []
        self.targets = []

        self.set_augmenter()

        self.image_ids = self.dataset.train_ids
        self.augmentation = True
        if self.mode == "val":
            self.image_ids = self.dataset.val_ids
            self.augmentation = False
        if self.mode == "test":
            self.image_ids = self.dataset.test_ids
            self.augmentation = False
        np.random.shuffle(self.image_ids)

    def reset(self):
        self.images.clear()
        self.targets.clear()

    def flow(self):
        b = 0
        i = 0

        while True:
            image_id = self.image_ids[i]
            i += 1
            if i >= len(self.image_ids):
                i = 0
                np.random.shuffle(self.image_ids)

            img = self.dataset.load_image(image_id)
            masks = self.dataset.load_masks(image_id)

            if self.augmentation:
                img, masks = self.augment_image(img, masks)
                if self.config.RANDOM_BRIGHTNESS > 0:
                    img = self.random_brightness(img)
            
            self.images.append(img)
            self.targets.append(masks)

            b += 1

            if b >= self.config.total_batchsize:
                inputs = np.asarray(self.images, dtype=np.float32)
                inputs = inputs / 255.0
                outputs = np.asarray(self.targets, dtype=np.float32)
                yield inputs, outputs
                b = 0
                self.reset()

    def set_augmenter(self):
        factors = []
        if self.config.RANDOM_HFLIP:
            factors.append(iaa.Fliplr(0.5))
        if self.config.RANDOM_VFLIP:
            factors.append(iaa.Flipud(0.5))
        affine = iaa.Affine(
            scale={
                "x": (1.0 - self.config.RANDOM_SCALE, 1.0 + self.config.RANDOM_SCALE),
                "y": (1.0 - self.config.RANDOM_SCALE, 1.0 + self.config.RANDOM_SCALE)},
            translate_percent={
                "x": (-self.config.RANDOM_SHIFT, self.config.RANDOM_SHIFT),
                "y": (-self.config.RANDOM_SHIFT, self.config.RANDOM_SHIFT)},
            rotate=(-self.config.RANDOM_ROTATE, self.config.RANDOM_ROTATE),
            shear=(-self.config.RANDOM_SHEAR, self.config.RANDOM_SHEAR)
        )
        factors.append(affine)
        if self.config.RANDOM_BLUR > 0:
            factors.append(iaa.GaussianBlur((0.0, self.config.RANDOM_BLUR)))
        if self.config.RANDOM_NOISE > 0:
            factors.append(iaa.AdditiveGaussianNoise(0, (0, self.config.RANDOM_NOISE)))
        # if self.config.RANDOM_BRIGHTNESS > 0:
        #     factors.append(iaa.MultiplyBrightness((
        #         1.0 - self.config.RANDOM_BRIGHTNESS,
        #         1.0 + self.config.RANDOM_BRIGHTNESS)))
        if self.config.RANDOM_CONTRAST > 0:
            factors.append(iaa.LogContrast(gain=(
                1.0 - self.config.RANDOM_CONTRAST,
                1.0 + self.config.RANDOM_CONTRAST)))
        self.augseq = iaa.Sequential(factors, random_order=True)

    def random_brightness(self, img):
        gain = random.uniform(1.0 - self.config.RANDOM_BRIGHTNESS,
                              1.0 + self.config.RANDOM_BRIGHTNESS)
        ret = img.astype(float) * gain
        ret = ret.clip(0.0, 255.0).astype(np.uint8)
        return ret

    def _hook(self, images, augmenter, parents, default):
        """Determines which augmenters to apply to masks."""
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                       "Fliplr", "Flipud", "CropAndPad",
                       "Affine", "PiecewiseAffine"]
        return augmenter.__class__.__name__ in MASK_AUGMENTERS
    

    def augment_image(self, img, masks):
        det = self.augseq.to_deterministic()
        img = det.augment_image(img)
        # only apply mask augmenters to masks
        res = []
        for class_id in range(masks.shape[2]):
            m = masks[:, :, class_id]
            m = det.augment_image(m, hooks=imgaug.HooksImages(activator=self._hook))
            res.append(m)
        res = np.stack(res, axis=2)
        return img, res

