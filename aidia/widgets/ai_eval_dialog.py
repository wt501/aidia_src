
import os
import logging
import cv2
import glob
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from qtpy import QtCore, QtWidgets, QtGui

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from aidia import qt
from aidia import utils
from aidia import __appname__, aidia_logger
from aidia import HOME_DIR, CLS, DET, SEG, LABEL_COLORMAP
from aidia.ai.config import AIConfig
from aidia.ai.dataset import Dataset
from aidia.ai.test import TestModel
from aidia.ai.det import DetectionModel
from aidia.ai.seg import SegmentationModel
from aidia.widgets import ImageWidget
from aidia import image
from aidia.dicom import DICOM, is_dicom

tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class AIEvalDialog(QtWidgets.QDialog):

    aiRunning = QtCore.Signal(bool)

    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint
                            | QtCore.Qt.WindowCloseButtonHint
                            | QtCore.Qt.WindowMaximizeButtonHint
                            )
        self.setWindowTitle(self.tr("AI Evaluation"))

        self.setMinimumSize(QtCore.QSize(1200, 800))

        self._layout = QtWidgets.QGridLayout()

        self._dataset_layout = QtWidgets.QVBoxLayout()
        self._dataset_widget = QtWidgets.QWidget()
        self._results_layout = QtWidgets.QVBoxLayout()
        self._results_widget = QtWidgets.QWidget()
        self._images_layout = QtWidgets.QVBoxLayout()
        self._images_widget = QtWidgets.QWidget()

        self.dataset_dir = ""
        self.target_name = ""
        self.log_dir = ""
        self.prev_dir = ""
        self.class_names = []
        self.task = None

        # self.fig, self.axes = plt.subplots(1, 2, figsize=(20, 8))
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        # plt.subplots_adjust(wspace=0.5)
        plt.rcParams["font.size"] = 15

        self.error_flags = {}
        self.input_fields = []
        self.left_row = 0
        self.right_row = 0

        # name
        self.tag_name = QtWidgets.QLabel(self.tr("Name"))
        self.input_name = QtWidgets.QComboBox()
        self.input_name.setMinimumWidth(200)
        def _validate(text):
            logdir = os.path.join(self.dataset_dir, "data", text)
            config_path = os.path.join(logdir, "config.json")
            dataset_path = os.path.join(logdir, "dataset.json")
            if (os.path.exists(config_path) and
                os.path.exists(dataset_path)):
                self._set_ok(self.tag_name)
                self.log_dir = logdir
            else:
                self._set_error(self.tag_name)
        self.input_name.currentTextChanged.connect(_validate)
        self._add_basic_params(self.tag_name, self.input_name)

        # select results box
        # self.tag_class = QtWidgets.QLabel(self.tr("Select Class"))
        # self.input_class = QtWidgets.QComboBox()
        # self.input_class.setMinimumWidth(150)
        # self.input_class.currentIndexChanged.connect(self.update_class_choice)
        # self._add_basic_params(self.tag_class, self.input_class)
        # self.input_class.setEnabled(False)

        ### add result fields ###
        title_dataset = qt.head_text(self.tr("Results"))
        title_dataset.setMaximumHeight(30)
        title_dataset.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter)
        self._results_layout.addWidget(title_dataset)

        self.text_results = QtWidgets.QLabel()
        self._results_layout.addWidget(self.text_results)

        ### add buttons ###
        # evaluate button
        self.button_eval = QtWidgets.QPushButton(self.tr("Evaluate"))
        self.button_eval.clicked.connect(self.evaluate)
        row = max(self.left_row, self.right_row)
        self._layout.addWidget(self.button_eval, row, 1, 1, 4)
        row += 1

        # predict button
        self.button_pred = QtWidgets.QPushButton(self.tr("Predict"))
        self.button_pred.setToolTip(self.tr(
            """Predict images in your directory."""
        ))
        self.button_pred.clicked.connect(self.predict_from_directory)
        self._layout.addWidget(self.button_pred, row, 1, 1, 4)
        row += 1

        # figure area
        self.image_widget = ImageWidget(self)
        self._layout.addWidget(self.image_widget, row, 1, 1, 4)
        row += 1

        # progress bar
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self._layout.addWidget(self.progress, row, 1, 1, 4)
        row += 1

        # status
        self.text_status = QtWidgets.QLabel()
        self.text_status.setMaximumHeight(50)
        self._layout.addWidget(self.text_status, row, 1, 1, 4)

        ### add dataset information ###
        # title
        title_dataset = qt.head_text(self.tr("Dataset Information"))
        title_dataset.setMaximumHeight(30)
        title_dataset.setAlignment(QtCore.Qt.AlignTop)
        self._dataset_layout.addWidget(title_dataset)

        # dataset information
        self.text_dataset = QtWidgets.QLabel()
        self.text_dataset.setAlignment(QtCore.Qt.AlignLeading)
        self._dataset_layout.addWidget(self.text_dataset)

        ### add preditcs images ###
        self.iw1 = ImageWidget(self)
        self.iw2 = ImageWidget(self)
        self.iw3 = ImageWidget(self)
        self.iw4 = ImageWidget(self)
        self.iw5 = ImageWidget(self)

        self._images_layout.addWidget(self.iw1)
        self._images_layout.addWidget(self.iw2)
        self._images_layout.addWidget(self.iw3)
        self._images_layout.addWidget(self.iw4)
        self._images_layout.addWidget(self.iw5)

        ### set layouts ###
        self._results_widget.setLayout(self._results_layout)
        self._dataset_widget.setLayout(self._dataset_layout)
        self._images_widget.setLayout(self._images_layout)

        self._layout.addWidget(self._results_widget, 0, 3, self.left_row, 2)
        self._layout.addWidget(self._dataset_widget, 0, 0, row + 1, 1)
        self._layout.addWidget(self._images_widget, 0, 5, row + 1, 1)

        self.setLayout(self._layout)

        # connect AI evaluation thread
        self.ai = AIEvalThread(self)
        self.ai.notifyMessage.connect(self.update_status)
        self.ai.datasetInfo.connect(self.update_dataset)
        self.ai.resultsList.connect(self.update_results)
        self.ai.predictList.connect(self.update_images)
        self.ai.progressValue.connect(self.update_progress)
        self.ai.finished.connect(self.ai_finished)

        # connect AI prediction thread
        self.ai_pred = AIPredThread(self)
        self.ai_pred.notifyMessage.connect(self.update_status)
        self.ai_pred.progressValue.connect(self.update_progress)
        self.ai_pred.finished.connect(self.ai_pred_finished)

        self.text_status.setText(self.tr("Ready"))
    
    def popup(self, dirpath):
        self.dataset_dir = dirpath
        self.setWindowTitle(self.tr("AI Evaluation - {}").format(dirpath))

        if not self.ai.isRunning():
            self.reset_state()

        # pickup log directories
        data_dir = os.path.join(dirpath, "data")
        if os.path.exists(data_dir):
            targets = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
            if len(targets):
                self.input_name.addItems(targets)
                self.enable_all()
                # if self.input_class.count() == 0:
                    # self.input_class.setEnabled(False)
            else:
                self.disable_all()
        else:
            self.disable_all()

        self.exec_()

    def ai_finished(self):
        """Call back function when AI thread finished."""
        self.enable_all()

        # save all figures
        # for i in range(len(self.class_names)):
        #     self.update_figure(i)

        # self.input_class.addItems(self.class_names)
        # self.input_class.setCurrentIndex(0)
        # self.input_class.setEnabled(True)

        self.aiRunning.emit(False)
    
    def ai_pred_finished(self):
        self.enable_all()
        self.aiRunning.emit(False)

    def disable_all(self):
        for x in self.input_fields:
            x.setEnabled(False)
        self.button_eval.setEnabled(False)
        self.button_pred.setEnabled(False)
    
    def enable_all(self):
        for x in self.input_fields:
            x.setEnabled(True)
        self.button_eval.setEnabled(True)
        self.button_pred.setEnabled(True)

    def closeEvent(self, event):
        self.input_name.clear()
        
    def showEvent(self, event):
        if self.ai.isRunning():
            self.disable_all()

    def update_images(self, images):
        self.iw1.loadPixmap(images[0])
        self.iw2.loadPixmap(images[1])
        self.iw3.loadPixmap(images[2])
        self.iw4.loadPixmap(images[3])
        self.iw5.loadPixmap(images[4])

    def _plt2img(self):
        self.fig.canvas.draw()
        data = self.fig.canvas.tostring_rgb()
        w, h = self.fig.canvas.get_width_height()
        c = len(data) // (w * h)
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)

    def update_dataset(self, value):
        dataset_num = value["dataset_num"]
        num_images = value["num_images"]
        num_shapes = value["num_shapes"]
        num_classes = value["num_classes"]
        num_train = value["num_train"]
        num_val = value["num_val"]
        num_test = value["num_test"]
        class_names = value["class_names"]
        num_per_class = value["num_per_class"]
        train_per_class = value["train_per_class"]
        val_per_class = value["val_per_class"]
        test_per_class = value["test_per_class"]
        self.train_steps = value["train_steps"]
        self.val_steps = value["val_steps"]

        labels_info = [self.tr("[*] labels (all|train|val|test)")]
        for i in range(num_classes):
            name = class_names[i]
            n = num_per_class[i]
            n_train = train_per_class[i]
            n_val = val_per_class[i]
            n_test = test_per_class[i]
            labels_info.append(f"[{i}] {name} ({n} | {n_train} | {n_val} | {n_test})")
        labels_info = "\n".join(labels_info)

        text = []
        text.append(self.tr("Dataset Number: {}").format(dataset_num))
        text.append(self.tr("Number of Data: {}").format(num_images))
        text.append(self.tr("Number of Train: {}").format(num_train))
        text.append(self.tr("Number of Validation: {}").format(num_val))
        text.append(self.tr("Number of Test: {}").format(num_test))
        if value.get("num_train_subdir") is not None:
            text.append(self.tr("Number of Train Directories: {}").format(value["num_train_subdir"]))
            text.append(self.tr("Number of Validation Directories: {}").format(value["num_val_subdir"]))
            text.append(self.tr("Number of Test Directories: {}").format(value["num_test_subdir"]))
        text.append(self.tr("Train Steps: {}").format(self.train_steps))
        text.append(self.tr("Validation Steps: {}").format(self.val_steps))
        text.append(self.tr("Number of Shapes: {}").format(num_shapes))
        text.append(self.tr("Class Information:\n{}").format(labels_info))
        text = "\n".join(text)
        self.text_dataset.setText(text)

        # write dataset information
        with open(os.path.join(self.log_dir, "dataset_info.txt"), mode="w", encoding="utf-8") as f:
            f.write(text)
    
    def update_progress(self, value):
        self.progress.setValue(value)

    def update_status(self, value):
        self.text_status.setText(str(value))

    def update_results(self, value):
        self.ax.clear()
        if self.task == DET:
            mAP = value["mAP50"]
            text = f"mAP50: {mAP}"
            self.text_results.setText(text)
            utils.save_dict_to_excel(value, os.path.join(self.log_dir, "eval.xlsx"))
        elif self.task == SEG:
            acc = value["Accuracy"]
            precision = value["Precision"]
            recall = value["Recall"]
            f1 = value["F1"]
            cm = value["ConfusionMatrix"]
            cm_disp = value["ConfusionMatrixPlot"]
            text = ""
            text += f"Accuracy: {acc:.6f}\n"
            text += f"Precision: {precision:.6f}\n"
            text += f"Recall (Sensitivity): {recall:.6f}\n"
            text += f"F1: {f1:.6f}\n"
            self.text_results.setText(text)

            save_dict = {
                "Accuracy": [acc],
                "Precision": [precision],
                "Recall (Sensitivity)": [recall],
                "F1": [f1],
            }
            utils.save_dict_to_excel(save_dict, os.path.join(self.log_dir, "eval.xlsx"))

            cm_disp.plot(ax=self.ax)
            self.image_widget.loadPixmap(self._plt2img())
            filename = os.path.join(self.log_dir, "confusion_matrix.png")
            # if not os.path.exists(filename):
            self.fig.savefig(filename)
        
    def update_class_choice(self, index):
        pass

    def _add_basic_params(self, tag:QtWidgets.QLabel, widget, right=False, reverse=False, custom_size=None):
        self.error_flags[tag.text()] = 0
        self.input_fields.append(widget)
        row = self.left_row
        pos = [1, 2]
        align = [QtCore.Qt.AlignRight, QtCore.Qt.AlignLeft]
        h, w = (1, 1)
        if right:
            row = self.right_row
            pos = [3, 4]
        if reverse:
            pos = pos[::-1]
            align = align[::-1]
        if custom_size:
            h = custom_size[0]
            w = custom_size[1]
        self._layout.addWidget(tag, row, pos[0], h, w, alignment=align[0])
        self._layout.addWidget(widget, row, pos[1], h, w, alignment=align[1])
        if right:
            self.right_row += h
        else:
            self.left_row += h

    def _set_error(self, tag:QtWidgets.QLabel):
        tag.setStyleSheet("QLabel{ color: red; }")
        self.error_flags[tag.text()] = 1
    
    def _set_ok(self, tag:QtWidgets.QLabel):
        tag.setStyleSheet("QLabel{ color: black; }")
        self.error_flags[tag.text()] = 0
                
    def create_input_field(self, size):
        l = QtWidgets.QLineEdit()
        l.setAlignment(QtCore.Qt.AlignRight)
        l.setMaximumWidth(size)
        l.setMinimumWidth(size)
        return l

    def reset_state(self):
        # self.input_class.clear()
        self.general_results = []
        self.results = []
        self.metrics_dict = {}
        self.progress.setValue(0)
        self.text_dataset.clear()
        self.text_results.clear()
        self.image_widget.clear()
        self.iw1.clear()
        self.iw2.clear()
        self.iw3.clear()
        self.iw4.clear()
        self.iw5.clear()

    def evaluate(self):
        error = sum(self.error_flags.values())
        if error > 0:
            self.text_status.setText(self.tr("Please check parameters."))
            return
        
        config_path = os.path.join(self.log_dir, "config.json")
        if not os.path.exists(config_path):
            self.text_status.setText(self.tr("Config file was not found."))
            return

        dataset_path = os.path.join(self.log_dir, "dataset.json")
        if not os.path.exists(dataset_path):
            self.text_status.setText(self.tr("Dataset file was not found."))
            return

        self.disable_all()
        self.reset_state()

        config = AIConfig(dataset_dir=self.dataset_dir)
        config.load(config_path)

        self.task = config.TASK
        self.class_names = config.LABELS.copy()
        if config.TASK == SEG:
            self.class_names.insert(0, "background")
        self.class_names.insert(0, "all")

        self.ai.set_config(config)
        self.ai.start()
        self.aiRunning.emit(True)
    
    def predict_from_directory(self):
        error = sum(self.error_flags.values())
        if error > 0:
            self.text_status.setText(self.tr("Please check parameters."))
            return

        if self.task not in [SEG]:
            self.text_status.setText(self.tr("Not implemented function."))
            return
        
        config_path = os.path.join(self.log_dir, "config.json")
        if not os.path.exists(config_path):
            self.text_status.setText(self.tr("Config file was not found."))
            return
        
        onnx_path = os.path.join(self.log_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            self.text_status.setText(self.tr("The ONNX model was not found."))
            return
                
        opendir = HOME_DIR
        if self.prev_dir and os.path.exists(self.prev_dir):
            opendir = self.prev_dir

        target_path = str(QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("{} - Open Directory".format(__appname__)),
            opendir,
            QtWidgets.QFileDialog.ShowDirsOnly |
            QtWidgets.QFileDialog.DontResolveSymlinks))
        target_path = target_path.replace("/", os.sep)
        if not target_path:
            return
        
        if not len(os.listdir(target_path)):
            self.text_status.setText(self.tr("The Directory is empty."))
            return
        
        self.text_status.setText(self.tr("Processing..."))

        self.prev_dir = target_path
        self.disable_all()
        self.progress.setValue(0)
        # self.reset_state()

        config = AIConfig(self.dataset_dir)
        config.load(config_path)
        self.task = config.TASK

        self.ai_pred.set_params(config, target_path, onnx_path)
        self.ai_pred.start()
        self.aiRunning.emit(True)

    
class AIEvalThread(QtCore.QThread):

    resultsList = QtCore.Signal(dict)
    # resultsList = QtCore.Signal(list)
    notifyMessage = QtCore.Signal(str)
    datasetInfo = QtCore.Signal(dict)
    predictList = QtCore.Signal(list)
    progressValue = QtCore.Signal(int)

    def __init__(self, parent):
        super().__init__(parent)

    def set_config(self, config:AIConfig):
        self.config = config
    
    def run(self):
        model = None
        if self.config.TASK == CLS:
            raise NotImplementedError
        elif self.config.TASK == DET:
            model = DetectionModel(self.config)
        if self.config.TASK == SEG:
            model = SegmentationModel(self.config)
        
        if model is None:
            self.notifyMessage.emit(self.tr("Model error. Terminated."))
            return
        
        self.notifyMessage.emit(self.tr("Data loading..."))
        try:
            model.load_dataset()
        except Exception as e:
            self.notifyMessage.emit(self.tr("Failed to load dataset."))
            aidia_logger.error(e, exc_info=True)
            return
        
        if isinstance(model.dataset, Dataset):
            _info_dict = {
                "dataset_num": model.dataset.dataset_num,
                "num_images": model.dataset.num_images,
                "num_shapes": model.dataset.num_shapes,
                "num_classes": model.dataset.num_classes,
                "num_per_class": model.dataset.num_per_class,
                "num_train": model.dataset.num_train,
                "num_val": model.dataset.num_val,
                "num_test": model.dataset.num_test,
                "class_ids": model.dataset.class_ids,
                "class_names": model.dataset.class_names,
                "train_per_class": model.dataset.train_per_class,
                "val_per_class": model.dataset.val_per_class,
                "test_per_class": model.dataset.test_per_class,
                "train_steps": model.dataset.train_steps,
                "val_steps": model.dataset.val_steps
            }
            if self.config.SUBMODE and self.config.DIR_SPLIT:
                _info_dict["num_subdir"] = model.dataset.num_subdir
                _info_dict["num_train_subdir"] = model.dataset.num_train_subdir
                _info_dict["num_val_subdir"] = model.dataset.num_val_subdir
                _info_dict["num_test_subdir"] = model.dataset.num_test_subdir
            self.datasetInfo.emit(_info_dict)

        self.notifyMessage.emit(self.tr("Model building..."))
        model.build_model(mode="test")

        self.notifyMessage.emit(self.tr("Generate test result images..."))

        save_dir = os.path.join(self.config.log_dir, "test_preds")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        n = model.dataset.num_test
        predicts = []
        for i in range(n):
            image_id = model.dataset.test_ids[i]
            img_path = model.dataset.image_info[image_id]["path"]
            name = os.path.splitext(os.path.basename(img_path))[0]
            try:
                result_img = model.predict_by_id(image_id)
            except FileNotFoundError as e:
                self.notifyMessage.emit(self.tr("Error: {} was not found.").format(img_path))
                return
            save_path = os.path.join(save_dir, f"{name}.png")
            image.imwrite(result_img, save_path)
            # update latest 5 images to the widget
            if len(predicts) <= 5:
                w = self.config.image_size[1]
                if self.config.TASK in [SEG]:
                    result_img = result_img[:, w:w*2]
                predicts.append(result_img)
            else:
                self.predictList.emit(predicts)
                predicts = []
            # update progress bar
            self.progressValue.emit(int(i / n * 100))
        self.progressValue.emit(0)

        self.notifyMessage.emit(self.tr("Convert model to ONNX..."))
        model.convert2onnx()
    
        self.notifyMessage.emit(self.tr("Evaluating..."))
        try:
            results = model.evaluate(cb_widget=self)
        except Exception as e:
            self.notifyMessage.emit(self.tr("Failed to evaluate."))
            aidia_logger.error(e, exc_info=True)
            return
        self.resultsList.emit(results)

        self.progressValue.emit(0)
        self.notifyMessage.emit(self.tr("Done."))


class AIPredThread(QtCore.QThread):

    notifyMessage = QtCore.Signal(str)
    progressValue = QtCore.Signal(int)

    def __init__(self, parent):
        super().__init__(parent)

    def set_params(self, config, target_path, onnx_path):
        self.config = config
        self.target_path = target_path
        self.onnx_path = onnx_path
    
    def run(self):
        savedir = os.path.join(self.target_path, "AI_results")
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        n = len(os.listdir(self.target_path))
        model = InferenceSession(self.onnx_path)

        # extensions = [".{}".format(fmt.data().decode("ascii").lower())
        #     for fmt in QtGui.QImageReader.supportedImageFormats()]
        for i, file_path in enumerate(glob.glob(os.path.join(self.target_path, "*"))):
            img = image.read_image(file_path)
            # if is_dicom(file_path) or utils.extract_ext(file_path) == ".dcm":
            #     dicom_data = DICOM(file_path)
            #     img = dicom_data.load_image()
            #     img = image.dicom_transform(
            #         img,
            #         dicom_data.wc,
            #         dicom_data.ww,
            #         dicom_data.bits
            #     )
            #     img = image.convert_dtype(img)
            #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # elif utils.extract_ext(file_path) in extensions:
            #     img = image.imread(file_path)
            if img is None:
                continue
            
            self.notifyMessage.emit(f"{i} / {n} - {file_path}")
            name = utils.get_basename(file_path)
            
            if self.config.TASK == SEG:
                img = cv2.resize(img, self.config.image_size)
                inputs = image.preprocessing(img)
                input_name = model.get_inputs()[0].name
                result = model.run([], {input_name: inputs})[0][0]
                result_img = image.mask2merge(img, result, self.config.LABELS)
                save_path = os.path.join(savedir, f"{name}.png")
                image.imwrite(result_img, save_path)
            else:
                raise NotImplementedError
            
            self.progressValue.emit(int(i / n * 100))

        self.progressValue.emit(0)
        self.notifyMessage.emit(self.tr("Saved result images to {}").format(savedir))