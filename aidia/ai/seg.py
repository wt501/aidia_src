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

        custom_metrics = ["binary_accuracy"]

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
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=custom_metrics)
        
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
        s = (self.dataset.num_test * self.config.image_size[0] * self.config.image_size[1],
             self.config.num_classes)
        for i, image_id in enumerate(self.dataset.test_ids):
            if cb_widget is not None:
                cb_widget.notifyMessage.emit(f"{i+1} / {self.dataset.num_test}")
                cb_widget.progressValue.emit(int((i+1) / self.dataset.num_test * 100))
            img = self.dataset.load_image(image_id)
            mask = self.dataset.load_masks(image_id)
            inputs = image.preprocessing(img, is_tensor=True)
            p = self.model.predict(inputs, verbose=0)[0]
            mask = mask[..., 1:] # exclude background
            p = p[..., 1:]
            y_true.append(mask)
            y_pred.append(p)
        y_true = np.array(y_true).reshape(s)
        y_pred = np.array(y_pred).reshape(s)

        # if cb_widget is not None:
        #     cb_widget.notifyMessage.emit("Calculating precision recall curve per class...")
        # pr_list = []
        # ap_list = []
        # for class_id in range(self.config.num_classes + 1):
        #     yt = y_true[..., class_id]
        #     yp = y_pred[..., class_id]
        #     pr = metrics.precision_recall_curve(yt, yp)
        #     ap = metrics.average_precision_score(yt, yp)
        #     pr_list.append(pr)
        #     ap_list.append(ap)

        if cb_widget is not None:
            cb_widget.notifyMessage.emit("Calculating confusion matrix...")
        yt = np.argmax(y_true, axis=1)
        yp = np.argmax(y_pred, axis=1)
        cm = metrics.confusion_matrix(yt, yp)
        cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                 display_labels=self.config.LABELS)
        acc = metrics.accuracy_score(yt, yp)
        precision = metrics.precision_score(yt, yp, average="macro")
        recall = metrics.recall_score(yt, yp, average="macro")
        f1 = metrics.f1_score(yt, yp, average="macro")
        res = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ConfusionMatrix": cm,
            "ConfusionMatrixPlot": cm_disp,
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

