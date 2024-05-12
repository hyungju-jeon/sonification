import sys

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QGridLayout
from pyqtgraph.Qt import QtCore, QtGui
from time import perf_counter


app = pg.mkQApp()
# -----------------------------Visualization on the wall-----------------------------
wall_parent_widget = gl.GLViewWidget()
wall_parent_widget.setGeometry(0, 0, 1920, 1200)
layout = QGridLayout()
wall_parent_widget.setLayout(layout)
monitor = QDesktopWidget().screenGeometry(0)
wall_parent_widget.move(monitor.left(), monitor.top())

wall_widget = gl.GLViewWidget(parent=wall_parent_widget)
wall_widget.setGeometry(0, 0, 1920, 1200)
wall_widget.opts["center"] = QtGui.QVector3D(-30, 0, 0)
wall_widget.opts["distance"] = 75
wall_widget.opts["fov"] = 90
wall_widget.opts["elevation"] = 0
wall_widget.opts["azimuth"] = 0

layout.addWidget(wall_widget, 1, 0)
wall_parent_widget.show()


image = np.random.rand(1000, 100) + 1
image[image < 0.5] = 0
raster_texture = pg.makeRGBA(image, levels=(0, 5))[0]
img = gl.GLImageItem(raster_texture)
scale_factor = [
    -50 / 1000,
    50 / 100,
    1,
]
img.scale(*scale_factor)
img.translate(
    +50 / 2 + 100 / 2,
    -50 / 2,
    -100 / 2 + 0,
)
wall_widget.addItem(img)

ptr = 0
lastTime = perf_counter()
fps = None


def update():
    global image
    image = np.roll(image, 1, axis=1)
    raster_texture = pg.makeRGBA(image, levels=(0, 2))[0]
    img.setData(raster_texture)
    app.processEvents()  ## force complete redraw for every plot


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)


app.exec()
