from qtpy import QtCore
from qtpy import QtWidgets

from aidia import qt
from aidia import utils
from aidia import __appname__

CLEAR, ERROR = 0, 1
NUM_MAX_LABELS = 10

class LabelSettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint)
        
        self.setWindowTitle(self.tr("Label Setting"))

        layout = QtWidgets.QVBoxLayout()

        self.label_def = []

        self.error = CLEAR

        # label definition
        self.label_def_input = QtWidgets.QTextEdit()
        self.label_def_input.textChanged.connect(self.parse_label)
        self.label_def_warning_text = QtWidgets.QLabel("")
        self.label_def_warning_text.setStyleSheet("color: red")
        layout.addWidget(self.label_def_input)
        layout.addWidget(self.label_def_warning_text)

        # accept and reject button
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        bb.button(bb.Ok).setIcon(qt.newIcon('done'))
        bb.button(bb.Cancel).setIcon(qt.newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)


    def validate(self):
        if self.error == ERROR:
            text = self.tr("Please check errors.")
            QtWidgets.QMessageBox.critical(
                self, self.tr("Error"),
                "<p>{}</p>".format(text))
            return
        else:
            self.accept()


    def popUp(self, label_def):
        self.label_def = label_def
        self.label_def_input.setText("\n".join(label_def))

        if self.exec_():
            return self.label_def
        else:
            return False

    
    def parse_label(self):
        text = self.label_def_input.toPlainText()
        text = text.strip().replace(" ", "")

        if len(text) == 0:
            self.error = ERROR
            self.label_def_warning_text.setText(self.tr("Empty labels."))
            return
        
        parsed = text.split("\n")
        res = [p for p in parsed if p != ""]
        res = list(dict.fromkeys(res))   # delete duplicates

        if utils.is_full_width(text):
            self.error = ERROR
            self.label_def_warning_text.setText(self.tr("Including 2 byte code."))
            return
        
        if len(res) > NUM_MAX_LABELS:
            self.error = ERROR
            self.label_def_warning_text.setText(self.tr("Please reduce labels to {} or below.").format(NUM_MAX_LABELS))
            return
        
        self.label_def = res
        self.error = CLEAR
        self.label_def_warning_text.clear()
        