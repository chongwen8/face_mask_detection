from PyQt5 import QtCore as qtc
import numpy as np
import cv2
from utils import prediction

class videoCaputre(qtc.QThread):
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)

    def __init__(self, status=0) -> None:
        super().__init__()
        self.run_flag = True
        self.status = status

    def run(self):
        cap = cv2.VideoCapture(self.status)

        while self.run_flag:
            ret, frame = cap.read()
            if ret is True:
                # try:
                if self.status != 0:
                    prediction_img = prediction(frame)
                else:
                    frame_flipped = cv2.flip(frame, 1)
                    prediction_img = prediction(frame_flipped)
                self.change_pixmap_signal.emit(prediction_img)

        cap.release()

    def stop(self):
        self.run_flag = False


