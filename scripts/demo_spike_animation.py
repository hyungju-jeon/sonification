import re
import sys
from matplotlib.pylab import f
from matplotlib.pyplot import sca
import numpy as np
from pandas import wide_to_long
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer
import asyncio
import threading


N = 50  # Length of the binary vector
R = 50  # Desired radius for the ripple effect


spike = [np.zeros((50, 1))]
SERVER_IP = "127.0.0.1"


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


class DiscAnimationWindow(pg.Qt.QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Disc Animation")
        self.central_widget = pg.Qt.QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.plot_widget = pg.PlotWidget()
        self.layout = pg.Qt.QtWidgets.QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.plot_widget)
        self.data = np.zeros(N)
        self.centroid_positions = [
            (np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(N)
        ]
        self.circles = []
        self.frame = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(1)  # Adjust the refresh rate as needed
        self.num_item = 0

    def update_animation(self):
        self.remove_marked_circles()
        if self.frame < 10000:
            self.data = np.random.uniform(0, 1, N)  # Update binary vector
            # self.data = np.zeros(N)
            # self.data[0] = 1
            for i, val in enumerate(self.data):
                if val > 0.99:
                    self.animate_point(i)
                    # Trigger animation event for each binary value of 1
            self.frame += 1
        else:
            self.timer.stop()

    def animate_point(self, index):
        self.remove_marked_circles()
        # print(f"Number of circles: {len(self.circles)}")
        # Perform ripple animation centered at the predefined position for the centroid
        pos = [
            self.centroid_positions[index]
        ]  # Use the predefined position for the centroid
        circle = pg.ScatterPlotItem()
        circle.setData(
            pos=pos,
            size=5,
            symbol="o",
            pen=pg.mkPen((255, 35, 35)),
            brush=(255, 0, 0, 0),
        )  # Initial small circle
        self.plot_widget.addItem(circle)
        self.circles.append((circle, 0))  # Store circle and its current radius

        QTimer.singleShot(
            1, lambda: self.expand_circle((circle, 0))
        )  # Start expanding the circle

    def expand_circle(self, circle):
        # print(f"Expanding circle {idx} out of {len(self.circles)}")
        # Get the circle and its radius
        item, radius = circle
        # Gradually increase the size of the circle until it reaches the desired radius (R)
        if radius < R:
            radius += 0.5
            item.setSize(radius)  # Multiply by 10 for better visualization
            # self.circles[idx] = (item, radius)  # Update the circle with the new radius
            QTimer.singleShot(
                1, lambda: self.expand_circle((item, radius))
            )  # Continue expanding
        else:
            # If the circle has reached the desired radius, mark it for removal
            # self.plot_widget.removeItem(circle)
            item.setPointsVisible(False)

        # Remove marked circles after completing expansion

    def remove_marked_circles(self):
        scatter_items = [
            item
            for item in self.plot_widget.items()
            if isinstance(item, pg.ScatterPlotItem)
        ]
        # # Iterate over the circles in reverse order to safely remove them
        for item in scatter_items:
            if not item.data["visible"]:
                self.plot_widget.removeItem(item)


class RippleAnimationWindow(pg.Qt.QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ripple Animation")
        self.central_widget = pg.Qt.QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-10, 10)
        self.plot_widget.setXRange(-10, 10)
        self.layout = pg.Qt.QtWidgets.QVBoxLayout(self.central_widget)
        # remove axis
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")

        self.layout.addWidget(self.plot_widget)
        self.data = np.zeros(N)
        self.centroid_positions = [
            (np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(N)
        ]
        self.circles = []
        self.frame = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(1)  # Adjust the refresh rate as needed

    def update_animation(self):
        self.remove_marked_circles()
        if self.frame < 10000:
            self.data = spike[0]  # Update binary vector
            spike[0] = np.zeros((50, 1))
            for i, val in enumerate(self.data):
                if val == 1:
                    self.animate_point(i)
            if np.sum(self.data) > 1:
                self.move_centroids()
                # Trigger animation event for each binary value of 1
            self.frame += 1
        else:
            self.timer.stop()

    def animate_point(self, index):
        # print(f"Number of circles: {len(self.circles)}")
        # Perform ripple animation centered at the predefined position for the centroid
        pos = [
            self.centroid_positions[index]
        ]  # Use the predefined position for the centroid

        ripple_bg = pg.ScatterPlotItem()
        ripple_bg.setData(
            pos=pos,
            size=10,
            symbol="o",
            pen=pg.mkPen(width=10, color=(0, 255, 65, 100)),
            brush=(255, 0, 0, 0),
        )  # Initial small circle
        self.plot_widget.addItem(ripple_bg)

        ripple = pg.ScatterPlotItem()
        ripple.setData(
            pos=pos,
            size=10,
            symbol="o",
            pen=pg.mkPen(width=3, color=(0, 255, 65)),
            brush=(255, 0, 0, 0),
        )  # Initial small circle
        self.plot_widget.addItem(ripple)

        zap_effect = pg.ScatterPlotItem()
        zap_effect.setData(
            pos=pos,
            size=4,
            symbol="o",
            brush=(255, 230, 230, 255),
        )  # Initial small circle
        self.plot_widget.addItem(zap_effect)

        QTimer.singleShot(
            1, lambda: self.expand_circle((ripple, 0))
        )  # Start expanding the circle
        QTimer.singleShot(
            1, lambda: self.expand_circle((ripple_bg, 0))
        )  # Start expanding the circle
        QTimer.singleShot(
            1, lambda: self.create_zap_effect(zap_effect)
        )  # Start expanding the circle

    def create_zap_effect(self, circle):
        QTimer.singleShot(100, lambda: circle.setPointsVisible(False))

    def expand_circle(self, circle):
        # print(f"Expanding circle {idx} out of {len(self.circles)}")
        # Get the circle and its radius
        item, radius = circle
        # Gradually increase the size of the circle until it reaches the desired radius (R)
        if radius < R:
            radius += 2
            item.setSize(radius)  # Multiply by 10 for better visualization
            current_pen = item.opts.get("pen")
            current_alpha = current_pen.color().getRgb()[-1]
            current_width = current_pen.width()
            item.setPen(
                pg.mkPen(
                    width=current_width,
                    color=(
                        current_pen.color().getRgb()[0],
                        current_pen.color().getRgb()[1],
                        current_pen.color().getRgb()[2],
                        current_alpha * 0.85,
                    ),
                )
            )
            # self.circles[idx] = (item, radius)  # Update the circle with the new radius
            QTimer.singleShot(
                10, lambda: self.expand_circle((item, radius))
            )  # Continue expanding
        else:
            # If the circle has reached the desired radius, mark it for removal
            # self.plot_widget.removeItem(circle)
            item.setPointsVisible(False)

        # Remove marked circles after completing expansion

    def remove_marked_circles(self):
        scatter_items = [
            item
            for item in self.plot_widget.items()
            if isinstance(item, pg.ScatterPlotItem)
        ]
        # # Iterate over the circles in reverse order to safely remove them
        for item in scatter_items:
            if not item.data["visible"]:
                self.plot_widget.removeItem(item)

    def move_centroids(self):
        # Calculate the centroid of activated points
        activated_positions = [
            self.centroid_positions[i] for i, val in enumerate(self.data) if val == 1
        ]

        unactivated_positions = [
            self.centroid_positions[i] for i, val in enumerate(self.data) if val == 0
        ]

        for i, center in enumerate(activated_positions):
            repulsion_force = np.array([0.0, 0.0])
            for j, other_center in enumerate(activated_positions):
                if i == j:
                    continue
                direction = np.array(other_center) - np.array(center)
                distance = np.linalg.norm(direction)
                if distance < 10:  # Adjust this threshold as needed
                    repulsion_force -= direction / (distance**2)
            self.centroid_positions[i] += (
                1 * repulsion_force
            )  # Adjust the repulsion force strength

        # Apply attraction force towards the centroid of activated points
        if activated_positions:
            centroid_of_activated = np.mean(activated_positions, axis=0)
            for i, center in enumerate(self.centroid_positions):
                direction = centroid_of_activated - np.array(center)

                self.centroid_positions[i] += (
                    0.01 * direction
                )  # Adjust the attraction force strength


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


if __name__ == "__main__":
    # asyncio.run(init_main())
    app = QApplication(sys.argv)
    window = RippleAnimationWindow()
    window.show()

    asyncio_thread = threading.Thread(target=asyncio_run)
    asyncio_thread.start()

    sys.exit(app.exec_())
