from qtpy import QtCore
from qtpy import QtWidgets

from aidia import qt
from aidia import utils
from aidia import __appname__
from aidia.qt import hline, head_text

LABEL, EPSILON = 0, 1


class SettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint)

        layout = QtWidgets.QVBoxLayout()
        params_layout = QtWidgets.QGridLayout()

        self.label_def = []
        self.approx_epsilon = 0.0

        self.error = [0, 0]

        # label definition
        self.label_def_input = QtWidgets.QTextEdit()
        self.label_def_input.textChanged.connect(self.parse_label)
        self.label_def_warning_text = QtWidgets.QLabel("")
        self.label_def_warning_text.setStyleSheet("color: red")
        layout.addWidget(head_text(self.tr("Label Definition")))
        layout.addWidget(self.label_def_input)
        layout.addWidget(self.label_def_warning_text)
        layout.addWidget(hline())

        # approx epsilon definition
        self.approx_epsilon_label = QtWidgets.QLabel(self.tr("Approximation Accuracy"))
        self.approx_epsilon_input = QtWidgets.QLineEdit()
        self.approx_epsilon_input.setAlignment(QtCore.Qt.AlignCenter)
        # self.approx_epsilon_input.setFixedWidth(50)
        def _validate(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 1.0:
                self.approx_epsilon = float(text)
                self.approx_epsilon_label.setStyleSheet("QLabel{ color: black; }")
                self.error[EPSILON] = 0
            else:
                self.approx_epsilon = 0.0
                self.approx_epsilon_label.setStyleSheet("QLabel{ color: red; }")
                self.error[EPSILON] = 1
        self.approx_epsilon_input.textChanged.connect(_validate)

        layout.addWidget(head_text(self.tr("Parameters")))
        params_layout.addWidget(self.approx_epsilon_label, 0, 0, QtCore.Qt.AlignRight)
        params_layout.addWidget(self.approx_epsilon_input, 0, 1, QtCore.Qt.AlignLeft)

        params_widget = QtWidgets.QWidget()
        params_widget.setLayout(params_layout)
        layout.addWidget(params_widget)

        # accept and reject button
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        bb.button(bb.Ok).setIcon(qt.newIcon('done'))
        bb.button(bb.Cancel).setIcon(qt.newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)


    def validate(self):
        if sum(self.error):
            text = self.tr("Please check errors.")
            QtWidgets.QMessageBox.critical(
                self, self.tr("Error"),
                "<p>{}</p>".format(text))
            return
        else:
            self.accept()


    def popUp(self, params_dict=None):
        label_def = params_dict["label_def"]
        approx_epsilon = params_dict["approx_epsilon"]

        self.label_def = label_def
        self.label_def_input.setText("\n".join(label_def))

        self.approx_epsilon = approx_epsilon
        self.approx_epsilon_input.setText(str(approx_epsilon))

        if self.exec_():
            ret_dict = {
                "label_def": self.label_def,
                "approx_epsilon": self.approx_epsilon
            }
            return ret_dict
        else:
            return None

    
    def parse_label(self):
        text = self.label_def_input.toPlainText()
        text = text.strip().replace(" ", "")

        if len(text) == 0:
            self.error[LABEL] = 1
            self.label_def_warning_text.setText(self.tr("Empty labels."))
            return
        
        parsed = text.split("\n")
        res = [p for p in parsed if p != ""]
        res = list(dict.fromkeys(res))   # delete duplicates

        if utils.is_full_width(text):
            self.error[LABEL] = 1
            self.label_def_warning_text.setText(self.tr("Including 2 byte code."))
            return
        
        if len(res) > 20:
            self.error[LABEL] = 1
            self.label_def_warning_text.setText(self.tr("Please reduce labels to 20 or below."))
            return
        
        self.label_def = res
        self.error[LABEL] = 0
        self.label_def_warning_text.clear()
        