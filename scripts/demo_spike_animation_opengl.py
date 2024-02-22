import asyncio
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pyqtgraph.Qt import QtCore, QtGui


spike = [np.zeros((50, 1))]
SERVER_IP = "127.0.0.1"

N = 50


def max_to_python_osc_handler(address, *args):
    """
    Handle OSC messages received from Max and update global variables accordingly.

    Parameters:
        address (str): The OSC address of the received message.
                       !! IMPORTANT !!
                       OSC address will be used as variable name. Make sure to match them!
        *args: Variable number of arguments representing the values of the OSC message.
    """
    global spike
    exec(address[1:] + "[0] = np.array(args)")
    # print(f"Received message {address}: {args}")


class Visualizer(object):
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.plot_widget = gl.GLViewWidget()
        self.plot_widget.opts["distance"] = 40
        self.plot_widget.setWindowTitle("pyqtgraph example: GLLinePlotItem")
        self.plot_widget.setGeometry(0, 110, 1024, 768)
        self.plot_widget.show()

        self.data = np.zeros(50)
        self.centroid_positions = [
            (np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(N)
        ]
        self.centroids = dict()
        self.traces = dict()

        # self.indicators, self.indicator_traces = self.draw_centroids()

        # for i, line in enumerate(self.y):
        #     y = np.array([line] * self.points)
        #     d = np.sqrt(self.x**2 + y**2)
        #     sine = 10 * np.sin(d + self.phase)
        #     pts = np.vstack([self.x, y, sine]).transpose()
        #     self.traces[i] = gl.GLLinePlotItem(
        #         pos=pts,
        #         color=pg.glColor((i, self.lines * 1.3)),
        #         width=(i + 1) / 10,
        #         antialias=True,
        #     )
        #     self.w.addItem(self.traces[i])

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def set_plotdata(self, name, points, color, width):
        self.traces[name].setData(pos=points, color=color, width=width)

    def update(self):
        stime = time.time()

        self.data = spike[0]
        spike[0] = np.zeros((50, 1))
        # Check and remove all marked circles (finished ripple animations)

        self.remove_marked_circles()

        # Update centroid location and replot centroid indicator
        # self.move_centroids()
        # self.update_centroid_indicator()

        # Update the binary vector and generate new spikes
        for i, val in enumerate(self.data):
            if val == 1:
                self.trigger_spike(i)

        print("{:.0f} FPS".format(1 / (time.time() - stime)))

    def trigger_spike(self, index):
        """
        Triggers a spike animation at the specified index.

        Args:
            index (int): The index of the spike to trigger.
        """

        pos = [self.centroid_positions[index]]

        # Add the ripple background glow element
        ripple_bg = gl.GLScatterPlotItem()
        ripple_bg.setData(pos=pos, size=30, color=(0, 1, 0.25, 0.8))
        self.plot_widget.addItem(ripple_bg)
        QTimer.singleShot(1, lambda: self.expand_circle((ripple_bg, 0)))

        # Add the ripple element
        ripple = gl.GLScatterPlotItem()
        ripple.setData(pos=pos, size=10, color=(0.5, 1, 0.5, 1))
        self.plot_widget.addItem(ripple)
        QTimer.singleShot(1, lambda: self.expand_circle((ripple, 0)))

        # Add the zapping effect element
        zap_effect = gl.GLScatterPlotItem()
        zap_effect.setData(
            pos=pos,
            size=10,
            color=(1, 1, 1, 0.8),
        )
        self.plot_widget.addItem(zap_effect)
        QTimer.singleShot(1, lambda: self.create_zap_effect(zap_effect))

    def create_zap_effect(self, circle):
        """
        Creates a zap effect for a circle by making it invisible after a delay.

        Args:
            circle (ScatterPlotItem): The circle to create the zap effect for.
        """
        QTimer.singleShot(100, lambda: circle.setData(color=(0, 0, 0, 0)))

    def expand_circle(self, circle_radius_tuple):
        """
        Expands a circle in the animation by gradually increasing its size.

        Args:
            circle (tuple): A tuple containing the circle item and its current radius.
        """
        circle, iter = circle_radius_tuple
        current_color = list(circle.color)
        current_color[-1] *= 0.80
        radius = circle.size + 3
        if iter < 30 and current_color[-1] > 0.05 and radius < 80:
            current_color = list(circle.color)
            current_color[-1] *= 0.90
            circle.setData(size=radius, color=current_color)
            QTimer.singleShot(10, lambda: self.expand_circle((circle, iter + 1)))
        else:
            circle.setData(color=(0, 0, 0, 0))
            # circle.setPointsVisible(False)

    def remove_marked_circles(self):
        """
        Removes marked circles from the animation.
        """
        scatter_items = [
            item
            for item in self.plot_widget.items
            if isinstance(item, gl.GLScatterPlotItem)
        ]
        print(len(scatter_items))
        for item in scatter_items:
            if item.color[-1] == 0:
                self.plot_widget.removeItem(item)

    def animation(self):
        timer = QTimer()
        timer.timeout.connect(self.update)
        timer.start(30)
        self.start()


async def init_main():
    # ----------------------------------------------------------- #
    # ------------------ OSC Receiver from Max ------------------ #
    dispatcher = Dispatcher()

    # pass the handlers to the dispatcher
    dispatcher.map("/spike", max_to_python_osc_handler)

    # you can have a default_handler for messages that don't have dedicated handlers
    def default_handler(address, *args):
        # print(f"No action taken for message {address}: {args}")
        pass

    dispatcher.set_default_handler(default_handler)

    # python-osc method for establishing the UDP communication with max
    server = AsyncIOOSCUDPServer(
        (SERVER_IP, 1123), dispatcher, asyncio.get_event_loop()
    )
    transport, protocol = await server.create_serve_endpoint()
    while True:
        try:
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            break


def asyncio_run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_main())


# Start event loop.
if __name__ == "__main__":
    asyncio_thread = threading.Thread(target=asyncio_run)
    asyncio_thread.start()

    v = Visualizer()
    v.animation()
