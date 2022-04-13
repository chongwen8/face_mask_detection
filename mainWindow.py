from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from videoCapture import videoCaputre
import numpy as np
import cv2
from utils import prediction

class mainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(qtg.QIcon('./images/icon.png'))
        self.setWindowTitle('Face mask recognition software')
        self.setFixedSize(600, 600)
        label = qtw.QLabel('<h2>Face Mask Recognition Application</h2>')
        self.camera_button = qtw.QPushButton('Open Camera', clicked=self.cameraButtonClick, checkable=True)
        self.browse_image_button = qtw.QPushButton('Browse Image', clicked=self.imageBrowse, checkable=True)
        self.open_video_button = qtw.QPushButton('Open Video', clicked=self.videoOpen, checkable=True)
        self.screen = qtw.QLabel()
        v_layout = qtw.QVBoxLayout()
        v_layout.addWidget(label)
        h_layout = qtw.QHBoxLayout()
        h_layout.addWidget(self.camera_button)
        h_layout.addWidget(self.browse_image_button)
        h_layout.addWidget(self.open_video_button)
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.screen)

        self.setLayout(v_layout)
        self.show()

    def resetBackGround(self):
        self.screen.clear()
        self.img = qtg.QPixmap(600, 480)
        self.img.fill(qtg.QColor('darkGrey'))
        self.screen.setPixmap(self.img)

    def cameraButtonClick(self):
        status = self.camera_button.isChecked()
        if status == True:
            self.camera_button.setText('Close Camera')

            self.capture = videoCaputre()
            self.capture.change_pixmap_signal.connect(self.updateImage)
            self.capture.start()

        else:
            self.camera_button.setText('Open Camera')
            self.capture.stop()
            self.resetBackGround()

    def imageBrowse(self):
        status = self.browse_image_button.isChecked()
        if status == True:
            fname = qtw.QFileDialog().getOpenFileName(self, "Open File", "/Users/chong", "Images (*.png *.xpm *.jpg *jpeg)")
            image_path = fname[0]
            img_array = cv2.imread(image_path)
            if img_array is not None:
                img = prediction(img_array)
                self.updateImage(img)
                self.browse_image_button.setText('Clear Image')
            else:
                status == False

        elif status == False:
            self.browse_image_button.setText('Browse Image')
            self.resetBackGround()

    def videoOpen(self):
        status = self.open_video_button.isChecked()
        if status == True:
            self.open_video_button.setText('Close Video')
            fname = qtw.QFileDialog().getOpenFileName(self, "Open File", "/Users/chong", "Video (*.mp4)")
            image_path = fname[0]
            self.capture = videoCaputre(status=image_path)
            self.capture.change_pixmap_signal.connect(self.updateImage)
            self.capture.start()

        else:
            self.open_video_button.setText('Open Video')
            self.capture.stop()
            self.resetBackGround()



    @qtc.pyqtSlot(np.ndarray)
    def updateImage(self, image_array):
        rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb_img.shape
        bytes_per_line = ch * w
        convertedImage = qtg.QImage(rgb_img.data,w,h,bytes_per_line,qtg.QImage.Format_RGB888)
        scaled_image = convertedImage.scaled(600,480,qtc.Qt.KeepAspectRatio)
        qt_img = qtg.QPixmap.fromImage(scaled_image)
        self.screen.setPixmap(qt_img)


