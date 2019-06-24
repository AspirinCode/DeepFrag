# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:42:01 2019

@author: hcji
"""

import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from DeepFragUI import Ui_MainWindow
from DeepFrag.utils import load_model, model_predict, plot_ms

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

class DeepFragApp(Ui_MainWindow):
    def __init__ (self, main_window):
        Ui_MainWindow.__init__(self)
        self.setupUi(main_window)
        self.pushButton.clicked.connect(self.predict_ms)
        self.pushButton.clicked.connect(self.plot_mol)
        
    def predict_ms(self):
        base = 'RIKEN_PlaSMA_'
        mode = self.ModInput.currentText()[0:3]
        eneg = self.EgyInput.currentText()[0:2]
        smi = self.SmiInput.toPlainText()
        model = load_model(base + mode + '_' + eneg)
        ms = model_predict(smi, model)
        # plot_ms(ms)
        self.F = MyFigure(width=3, height=2, dpi=100)
        self.F.axes.cla()
        self.F.axes.vlines(ms['mz'], np.zeros(ms.shape[0]), np.array(ms['intensity']), 'red') 
        self.F.axes.axhline(0, color='black')
        self.gridlayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridlayout.addWidget(self.F,0,1)
        self.gridlayout.deleteLater()
    
    def plot_mol(self):
        mol = Chem.MolFromSmiles(self.SmiInput.toPlainText())
        Draw.MolToFile(mol,'mol.png')
        pixmap = QPixmap('mol.png')
        self.MolView.setPixmap(pixmap)
        
        
class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MyFigure,self).__init__(self.fig) 
        self.axes = self.fig.add_subplot(111)
             
        


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    test = DeepFragApp(main_window)
    main_window.show()
    sys.exit(app.exec_())