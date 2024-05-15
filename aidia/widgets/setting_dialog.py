from qtpy import QtCore
from qtpy import QtWidgets

from aidia import qt
from aidia import __appname__
from aidia.qt import hline, head_text
from aidia import S_EPSILON, CLEAR, ERROR


class SettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint)

        layout = QtWidgets.QVBoxLayout()
        general_layout = QtWidgets.QGridLayout()

        self.approx_epsilon = 0.0

        self.error = {}

        # approx epsilon definition
        self.approx_epsilon_text = QtWidgets.QLabel(self.tr("Approximation Accuracy"))
        self.approx_epsilon_input = QtWidgets.QLineEdit()
        self.approx_epsilon_input.setAlignment(QtCore.Qt.AlignCenter)
        # self.approx_epsilon_input.setFixedWidth(50)
        def _validate(text):
            if text.replace(".", "", 1).isdigit() and 0.0 < float(text) < 1.0:
                self.approx_epsilon = float(text)
                self.approx_epsilon_text.setStyleSheet("QLabel{ color: black; }")
                self.error[S_EPSILON] = CLEAR
            else:
                self.approx_epsilon = 0.0
                self.approx_epsilon_text.setStyleSheet("QLabel{ color: red; }")
                self.error[S_EPSILON] = ERROR
        self.approx_epsilon_input.textChanged.connect(_validate)

        # set general settings layout
        layout.addWidget(head_text(self.tr("General Settings")))
        general_layout.addWidget(self.approx_epsilon_text, 0, 0, QtCore.Qt.AlignRight)
        general_layout.addWidget(self.approx_epsilon_input, 0, 1, QtCore.Qt.AlignLeft)
        general_widget = QtWidgets.QWidget()
        general_widget.setLayout(general_layout)
        layout.addWidget(general_widget)

        # accept and reject button
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        bb.button(bb.Ok).setIcon(qt.newIcon('done'))
        bb.button(bb.Cancel).setIcon(qt.newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)


    def validate(self):
        if sum(self.error.values()) > 0:
            text = self.tr("Please check errors.")
            QtWidgets.QMessageBox.critical(
                self, self.tr("Error"),
                "<p>{}</p>".format(text))
            return
        else:
            self.accept()


    def popUp(self, params_dict=None):
        approx_epsilon = params_dict[S_EPSILON]

        self.approx_epsilon = approx_epsilon
        self.approx_epsilon_input.setText(str(approx_epsilon))

        if self.exec_():
            return {
                    S_EPSILON: self.approx_epsilon
                }
        else:
            return False
