import os
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget, QFileDialog

from Libs.pipeline import Pipeline


class Widget(QWidget):

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("main.ui", self)
        self.pipeline = Pipeline()
        self.ui.pushButton.clicked.connect(self.single_image)
        self.ui.pushButton_2.clicked.connect(self.multiple_image)
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.hide()

    def single_image(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Open File', './', "*")
        file_name = str(file).split('/')[-1]
        try:
            result = self.pipeline.process(file)
            with_bboxes = self.pipeline.result['detector']['img_w_boxes']
            cv2.imshow('result', cv2.resize(with_bboxes, (320, 320)))
            self.ui.textBrowser.append(f'{file_name}: {result}\n')
        except Exception as e:
            self.ui.textBrowser.append(f'{file_name}: {e}\n')

    def multiple_image(self):
        stats = {
            'DEER': 0,
            'MUSKDEER': 0,
            'ROEDEER': 0,
            'NODEER': 0
        }
        self.ui.progressBar.show()
        self.ui.progressBar.value = 0
        os.makedirs('results', exist_ok=True)
        os.makedirs(os.path.join('results', 'DEER'), exist_ok=True)
        os.makedirs(os.path.join('results', 'MUSKDEER'), exist_ok=True)
        os.makedirs(os.path.join('results', 'ROEDEER'), exist_ok=True)
        os.makedirs(os.path.join('results', 'NODEER'), exist_ok=True)
        dir_path = QFileDialog.getExistingDirectory(self, 'Open Dir')
        dir_content = os.listdir(dir_path)
        dir_content_len = len(dir_content)
        if dir_content_len > 0:
            for ind, item in enumerate(dir_content):
                percent = int((ind + 1) / dir_content_len * 100)
                try:
                    self.ui.progressBar.setValue(percent)
                    result = self.pipeline.process(os.path.join(dir_path, item))
                    stats[result] += 1
                    self.ui.textBrowser.append(f'{item}: {result}\n')
                    shutil.copy(os.path.join(dir_path, item),
                                os.path.join('results', result, item))
                except Exception as e:
                    pass
            print(stats)
            plt.bar(list(stats.keys()), stats.values(), color='g')
            plt.show()
        self.ui.textBrowser.append('Done')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Widget()
    ex.show()
    app.exec_()
