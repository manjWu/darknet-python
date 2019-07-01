import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QImage, QPixmap
from processVideo import processVideo
import multiprocessing
import cv2

qtCreatorFile = "mainWindow.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.run_offline_button.clicked.connect(self.runOffline)

    def runOffline(self):
        filePath = int(self.filePath.toPlainText())
        with multiprocessing.Manager() as MG:  #重命名#
            result_list = MG.list()
            p1_img = multiprocessing.Process(target=processVideo, args=(result_list, filePath, 25,))  # 创建新进程1
            p2_show = multiprocessing.Process(target=self.refreshShow, args=(result_list,))  # 创建新进程2
            p1_img.start()
            p2_show.start()
            # 当处理程序结束后终止显示程序
            p1_img.join()
            # p2_show进程里是死循环，无法等待其结束，只能强行终止
            p2_show.terminate()
            print('Child process end.')

    def refreshShow(self,result_list):
        while(len(result_list)>2):
            # 提取图像的尺寸和通道, 用于将opencv下的image转换成Qimage
            height, width, channel = result_list[0].shape
            bytesPerLine = 3 * width
            qImg = QImage(result_list[0].data, width, height, bytesPerLine,
                               QImage.Format_RGB888).rgbSwapped()

            # 将Qimage显示出来
            self.result_video.setPixmap(QPixmap.fromImage(qImg))
            print("refreshShow...")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())