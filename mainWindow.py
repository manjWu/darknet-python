# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(754, 600)
        MainWindow.setStatusTip("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 731, 541))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.run_offline_button = QtWidgets.QPushButton(self.tab)
        self.run_offline_button.setGeometry(QtCore.QRect(590, 190, 113, 32))
        self.run_offline_button.setObjectName("run_offline_button")
        self.filePath = QtWidgets.QTextEdit(self.tab)
        self.filePath.setGeometry(QtCore.QRect(110, 30, 431, 21))
        self.filePath.setObjectName("filePath")
        self.FilePath_label = QtWidgets.QLabel(self.tab)
        self.FilePath_label.setGeometry(QtCore.QRect(40, 30, 71, 21))
        self.FilePath_label.setObjectName("FilePath_label")
        self.result_text = QtWidgets.QTextEdit(self.tab)
        self.result_text.setGeometry(QtCore.QRect(40, 440, 501, 51))
        self.result_text.setObjectName("result_text")
        self.stop_offline_button = QtWidgets.QPushButton(self.tab)
        self.stop_offline_button.setGeometry(QtCore.QRect(590, 230, 113, 32))
        self.stop_offline_button.setObjectName("stop_offline_button")
        self.result_video = QtWidgets.QLabel(self.tab)
        self.result_video.setGeometry(QtCore.QRect(50, 80, 491, 321))
        self.result_video.setObjectName("result_video")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 754, 22))
        self.menuBar.setNativeMenuBar(False)
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menuBar)
        self.actionsvs = QtWidgets.QAction(MainWindow)
        self.actionsvs.setObjectName("actionsvs")
        self.actionMenu = QtWidgets.QAction(MainWindow)
        self.actionMenu.setObjectName("actionMenu")
        self.actionNew_file = QtWidgets.QAction(MainWindow)
        self.actionNew_file.setObjectName("actionNew_file")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionNew_file)
        self.menuFile.addAction(self.actionQuit)
        self.menuBar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.actionQuit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tab.setStatusTip(_translate("MainWindow", "离线视频运行"))
        self.run_offline_button.setText(_translate("MainWindow", "运行"))
        self.FilePath_label.setText(_translate("MainWindow", "视频路径："))
        self.stop_offline_button.setText(_translate("MainWindow", "结束"))
        self.result_video.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Tab 1"))
        self.tab_2.setStatusTip(_translate("MainWindow", "在线运行"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionsvs.setText(_translate("MainWindow", "svs"))
        self.actionMenu.setText(_translate("MainWindow", "Menu"))
        self.actionNew_file.setText(_translate("MainWindow", "New file"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))

