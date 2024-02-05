from qtpy import QtWidgets, QtGui, QtCore

class ImageWidget(QtWidgets.QWidget):
    
    def __init__(self, parent, image=None):
        super().__init__(parent=parent)

        self._painter = QtGui.QPainter()
        self.pixmap = QtGui.QPixmap()
        if image is not None:
            self.loadPixmap(image)
        self.show()
    
    def loadPixmap(self, image):
        byte_per_line = image[0].nbytes
        h, w = image.shape[0:2]
        image = QtGui.QImage(image.flatten(), w, h, byte_per_line,
                            QtGui.QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(image)
        self.setMinimumHeight(self.pixmap.height() // 4)
        self.setMinimumWidth(self.pixmap.width() // 4)
        self.update()
    
    def clear(self):
        self.pixmap = QtGui.QPixmap()
        self.update()
    
    def paintEvent(self, event):
        p = self._painter
        p.begin(self)

        if self.pixmap.isNull():
            # return super().paintEvent(event)
            self.pixmap.fill()
        else:
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

            w = self.pixmap.width()
            h = self.pixmap.height()
            if w > 0 or h > 0:
                w_scale = self.width() / w
                h_scale = self.height() / h
                p.scale(w_scale, h_scale)

        p.drawPixmap(0, 0, self.pixmap)
        p.end()

    def resizeEvent(self, event):
        self.update()