import functools
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets



class LabelDialog(QtWidgets.QWidget):

    valueChanged = QtCore.Signal()

    def __init__(self, parent=None, label_def=None):

        super().__init__(parent)

        self.label_def = []
        if label_def:
            self.label_def = label_def

        self.button_size = 80
        self.line_break = 4

        self.label_list = []
        self.default_label_list = []
        self.label = ""
        self.default_label = ""
        self.prev_button = None
        self.b_dict = {}
        self.b2_dict = {}

        self.selected_text = QtWidgets.QLabel(self.tr("Selected Shape Labels"))
        self.default_text = QtWidgets.QLabel(self.tr("Default Labels"))

        self.layout = QtWidgets.QVBoxLayout()
        self.create_label_buttons()
        self.setLayout(self.layout)
        # self.setEnabled(False)
        self.disable_buttons()
    

    def enable_buttons(self):
        for b in self.b_dict.values():
            b.setEnabled(True)


    def disable_buttons(self):
        for b in self.b_dict.values():
            b.setEnabled(False)
    
    
    def create_label_buttons(self):
        # clear buttons
        if self.layout.count() > 0:
            for i in range(self.layout.count()):
                item = self.layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)

        # add buttons
        bb_list = []
        bb2_list = []
        num_labels = len(self.label_def)
        self.b_dict = {}
        self.b2_dict = {}
        n = (num_labels - 1) // self.line_break + 1
        for _ in range(n):
            bb = QtWidgets.QDialogButtonBox(QtCore.Qt.Horizontal)
            bb2 = QtWidgets.QDialogButtonBox(QtCore.Qt.Horizontal)
            # bb.clicked.connect(self.clicked_button)
            bb_list.append(bb)
            bb2_list.append(bb2)
        
        for i, l in enumerate(self.label_def):
            b = self.create_button(l, self.button_size)
            b.clicked.connect(functools.partial(self.set_label, l))
            self.b_dict[l] = b
            x = i // self.line_break
            bb_list[x].addButton(b, QtWidgets.QDialogButtonBox.ActionRole)
        
        for j, l in enumerate(self.label_def):
            b = self.create_button(l, self.button_size)
            b.clicked.connect(functools.partial(self.set_default_label, l))
            self.b2_dict[l] = b
            x = j // self.line_break
            bb2_list[x].addButton(b, QtWidgets.QDialogButtonBox.ActionRole)

        self.layout.addWidget(self.selected_text)
        for bb in bb_list:
            self.layout.addWidget(bb, alignment=QtCore.Qt.AlignLeft)

        self.layout.addWidget(self.default_text)
        
        for bb2 in bb2_list:
            self.layout.addWidget(bb2, alignment=QtCore.Qt.AlignLeft)
        

    def update_label_def(self, label_def):
        self.label_def = label_def
        self.create_label_buttons()


    def update(self, label=None):
        if label is None:
            # self.setEnabled(False)
            self.disable_buttons()
        else:
            # self.setEnabled(True)
            self.enable_buttons()
            if len(label) == 0:
                self.label_list = []
            else:
                self.label_list = label.split("_")

            # update button states by label
            for l in self.label_list:
                if l in self.label_def:
                    self.b_dict[l].setChecked(True)
            self.validate()
                        
    
    def validate(self):
        if len(self.label_list):
            self.label = self.label_list[0]
            if len(self.label_list) > 1:
                for l in self.label_list[1:]:
                    self.label += f"_{l}"
        else:
            self.label = ""
    

    def reset(self):
        self.label_list.clear()
        self.label = ""
        self.reset_button()
        # self.setEnabled(False)
        self.disable_buttons()


    def reset_button(self):
        for v in self.b_dict.values():
            v.setChecked(False)
    

    def clicked_button(self, button):
        if self.prev_button is None:
            self.prev_button = button
        elif self.prev_button == button:
            pass
        else:
            self.prev_button.setChecked(False)
            self.prev_button = button


    def create_button(self, button_label, size, color=None):
        button = QtWidgets.QPushButton('&{}'.format(button_label))
        button.setCheckable(True)
        button.setAutoDefault(False)
        # button.setMaximumWidth(size)
        # button.setMinimumWidth(size)
        if color:
            pal = QtGui.QPalette()
            pal.setColor(QtGui.QPalette.Button, QtGui.QColor(color))
            # pal.setColor(QtGui.QPalette.Window, color)
            button.setAutoFillBackground(True)
            button.setPalette(pal)
        return button


    def set_label(self, label):
        if label in self.label_list:
            self.label_list.remove(label)
        else:
            self.label_list.append(label)
        self.validate()
        self.valueChanged.emit()


    def set_default_label(self, label):
        if label in self.default_label_list:
            self.default_label_list.remove(label)
        else:
            self.default_label_list.append(label)
        self.default_label = "_".join(self.default_label_list)