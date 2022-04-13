import sys
from mainWindow import mainWindow
from PyQt5 import QtWidgets as qtw


if __name__ == "__main__":
    app = qtw.QApplication()
    mw = mainWindow()
    sys.exit(app.exec())