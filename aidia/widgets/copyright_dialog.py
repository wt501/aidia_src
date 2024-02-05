import cv2
import os.path as osp
from qtpy import QtCore
from qtpy import QtWidgets

from aidia.widgets import ImageWidget
from aidia.qt import head_text
from aidia import APP_DIR
from aidia.image import imread


class CopyrightDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.CustomizeWindowHint
                            | QtCore.Qt.WindowTitleHint
                            | QtCore.Qt.WindowCloseButtonHint)

        text = QtWidgets.QLabel("Copyright (C) 2021-2023 Kohei Torii.")
        text2 = QtWidgets.QLabel("""Copyright (C) 2021-2023 Kohei Torii.
Copyright (C) 2016-2018 Kentaro Wada.
Copyright (C) 2011 Michael Pitidis, Hussein Abdulwahid.

Aidia is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Aidia is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Aidia. If not, see <http://www.gnu.org/licenses/>.""")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(head_text("Copyright"))
        layout.addWidget(text)
        layout.addWidget(head_text("License"))
        layout.addWidget(text2)

        self.setLayout(layout)

    def popUp(self):
        self.exec_()
