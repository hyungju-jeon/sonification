import asyncio
import sys
import threading
import time
from matplotlib.pylab import f
import numpy as np
import pyqtgraph as pg

from pyqtgraph.Qt import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

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


class RippleAnimation(object):
    """
    A class for displaying a ripple animation.

    Attributes:
        app (QApplication): The QApplication instance.
        plot_widget (PlotWidget): The PlotWidget instance.
        data (ndarray): The binary vector representing the data.
        centroid_positions (list): The positions of the centroids.
        indicators (ScatterPlotItem): The centroid indicator items in the animation.
        indicator_traces (list): The centroid indicator trace items in the animation.
        frame (int): The current frame of the animation.

    Methods:
        __init__(): Initializes the RippleAnimationWindow.
        update_animation(): Updates the animation.
        animate_point(index): Animates a point in the animation.
        create_zap_effect(circle): Creates a zap effect for a circle.
        expand_circle(circle): Expands a circle in the animation.
        remove_marked_circles(): Removes marked circles from the animation. (Garbage collection)
        move_centroids(): Moves the centroids based on the activated points.
    """

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.central_widget = pg.Qt.QtWidgets.QWidget()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-10, 10)
        self.plot_widget.setXRange(-10, 10)
        self.plot_widget.setGeometry(0, 110, 1024, 768)
        # remove axis
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        self.plot_widget.show()

        self.data = np.zeros(N)
        self.centroid_positions = [
            (np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(N)
        ]
        self.indicators, self.indicator_traces = self.draw_centroids()
        self.velocity = np.zeros((N, 2))

        # Add function that will be called when the timer times out
        self.frame = 0

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def animation(self):
        timer = QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)
        self.start()

    def update(self):
        """
        Updates the animation by removing marked circles, updating the binary vector,
        animating points, moving centroids, and stopping the timer when the animation is complete.
        """
        stime = time.perf_counter_ns()
        self.data = spike[0]
        spike[0] = np.zeros((50, 1))

        # Check and remove all marked circles (finished ripple animations)
        self.remove_marked_circles()

        # Update centroid location and replot centroid indicator
        self.move_centroids()
        self.update_centroid_indicator()

        print(f"FPS : {(time.perf_counter_ns() - stime) / 1e6}")

        # Update the binary vector and generate new spikes
        for i, val in enumerate(self.data):
            if val == 1:
                self.trigger_spike(i)
        self.frame += 1

    def trigger_spike(self, index):
        """
        Triggers a spike animation at the specified index.

        Args:
            index (int): The index of the spike to trigger.
        """
        pos = [self.centroid_positions[index]]

        # Add the ripple background glow element
        ripple_bg = pg.ScatterPlotItem()
        ripple_bg.setData(
            pos=pos,
            size=10,
            symbol="o",
            pen=pg.mkPen(width=8, color=(0, 255, 65, 100)),
            brush=(255, 0, 0, 0),
        )
        self.plot_widget.addItem(ripple_bg)
        QTimer.singleShot(0, lambda: self.expand_circle((ripple_bg, 0)))

        # # Add the ripple element
        ripple = pg.ScatterPlotItem()
        ripple.setData(
            pos=pos,
            size=8,
            symbol="o",
            pen=pg.mkPen(width=3, color=(125, 255, 150, 255)),
            brush=(255, 0, 0, 0),
        )
        self.plot_widget.addItem(ripple)
        QTimer.singleShot(10, lambda: self.expand_circle((ripple, 0)))

        # Add the zapping effect element
        zap_effect = pg.ScatterPlotItem()
        zap_effect.setData(
            pos=pos,
            size=8,
            symbol="o",
            brush=(255, 230, 230, 255),
        )
        self.plot_widget.addItem(zap_effect)
        QTimer.singleShot(0, lambda: self.create_zap_effect(zap_effect))

    def create_zap_effect(self, circle):
        """
        Creates a zap effect for a circle by making it invisible after a delay.

        Args:
            circle (ScatterPlotItem): The circle to create the zap effect for.
        """
        QTimer.singleShot(30, lambda: circle.setPointsVisible(False))

    def expand_circle(self, circle_iter_tuple):
        """
        Expands a circle in the animation by gradually increasing its size.

        Args:
            circle (tuple): A tuple containing the circle item and its current radius.
        """
        circle, iter = circle_iter_tuple
        current_pen = circle.opts.get("pen")
        current_color = list(current_pen.color().getRgb())
        radius = circle.opts.get("size")

        if iter < 20 and current_color[-1] > 10 and radius < 40:
            circle.setSize(radius + 2)
            current_width = current_pen.width()
            current_color[-1] *= 0.85
            circle.setPen(
                pg.mkPen(
                    width=current_width,
                    color=current_color,
                )
            )
            QTimer.singleShot(10, lambda: self.expand_circle((circle, iter + 1)))
        else:
            circle.setPointsVisible(False)

    def remove_marked_circles(self):
        """
        Removes marked circles from the animation.
        """
        scatter_items = [
            item
            for item in self.plot_widget.items()
            if isinstance(item, pg.ScatterPlotItem)
        ]
        for item in scatter_items:
            if item.data.size > 0:
                if not all(item.data["visible"]):
                    self.plot_widget.removeItem(item)

    def draw_centroids(self):
        """
        Draws the centroid indicators in the animation.

        Returns:
            list: The centroid indicator items in the animation.
        """
        indicator = pg.ScatterPlotItem()
        indicator.setData(
            pos=self.centroid_positions,
            size=3,
            symbol="o",
            brush=(200, 200, 200, 200),
        )
        self.plot_widget.addItem(indicator)

        indicator_traces = []
        for i in range(N):
            indicator_trace = pg.PlotCurveItem()
            indicator_trace.setData(x=[0, 0], y=[0, 0])
            indicator_traces.append(indicator_trace)
            self.plot_widget.addItem(indicator_trace)
        return indicator, indicator_traces

    def update_centroid_indicator(self):
        """
        Updates the centroid indicator positions in the animation.
        """
        stime = time.perf_counter_ns()
        x, y = self.indicators.getData()
        # get first two columns of pos
        pos = np.column_stack((x, y))
        # if new position is out of bounds, reverse the velocity
        for i, center in enumerate(pos):
            self.velocity[i] = np.clip(self.velocity[i], -1, 1)
            if center[0] > 10 or center[0] < -10:
                self.velocity[i, 0] *= -1
            if center[1] > 10 or center[1] < -10:
                self.velocity[i, 1] *= -1
        new_pos = pos + self.velocity * 0.1
        for i, trace in enumerate(self.indicator_traces):
            trace.setData(
                x=[pos[i][0], new_pos[i][0]],
                y=[pos[i][1], new_pos[i][1]],
            )

        self.indicators.setData(pos=new_pos, skipFiniteCheck=True)
        self.centroid_positions = new_pos
        print(f"msec : {(time.perf_counter_ns() - stime) /1e9}")

    def move_centroids_jump(self):
        """
        Moves the centroids based on the activated points.
        """
        activated_positions = [
            (i, self.centroid_positions[i])
            for i, val in enumerate(self.data)
            if val == 1
        ]
        activated_index = [i for i, val in enumerate(self.data) if val == 1]

        unactivated_positions = [
            (i, self.centroid_positions[i])
            for i, val in enumerate(self.data)
            if val == 0
        ]

        if len(activated_positions) > 1:
            second_elements = np.array([t[1] for t in activated_positions])
            # Calculating the mean vector of the second elements
            mean_vector = np.mean(second_elements, axis=0)
            centroid_of_activated = mean_vector

            for i, center in activated_positions:
                direction = centroid_of_activated - np.array(center)
                self.centroid_positions[i] += 0.5 * direction

        for i, center in enumerate(self.centroid_positions):
            repulsion_force = np.array([0.0, 0.0])
            for j, other_center in enumerate(self.centroid_positions):
                if i == j or (i in activated_index and j in activated_index):
                    continue
                direction = np.array(other_center) - np.array(center)
                distance = np.linalg.norm(direction)
                if distance < 1:
                    repulsion_force -= direction / (distance**2)
            self.centroid_positions[i] += 0.01 * repulsion_force

    def move_centroids(self):
        """
        Moves the centroids based on the activated points.
        """
        activated_positions = [
            (i, self.centroid_positions[i])
            for i, val in enumerate(self.data)
            if val == 1
        ]
        for i, center in activated_positions:
            repulsion_force = np.array([0.0, 0.0])
            for j, other_center in activated_positions:
                if i == j:
                    continue
                direction = np.array(other_center) - np.array(center)
                distance = np.linalg.norm(direction)
                if distance < 5:
                    repulsion_force += direction / (distance**2)
            self.velocity[i] += 0.1 * repulsion_force

        for i, center in enumerate(self.centroid_positions):
            repulsion_force = np.array([0.0, 0.0])
            for j, other_center in enumerate(self.centroid_positions):
                if i == j:
                    continue
                direction = np.array(other_center) - np.array(center)
                distance = np.linalg.norm(direction)
                if distance < 3:
                    repulsion_force -= direction / (distance**2)
            self.velocity[i] += 0.1 * repulsion_force


class DiscAnimation(object):
    """
    A class for displaying a disc animation.

    Attributes:
        app (QApplication): The QApplication instance.
        plot_widget (PlotWidget): The PlotWidget instance.
        data (ndarray): The binary vector representing the data.
        centroid_positions (list): The positions of the centroids.
        indicators (ScatterPlotItem): The centroid indicator items in the animation.
        indicator_traces (list): The centroid indicator trace items in the animation.
        frame (int): The current frame of the animation.

    Methods:
        __init__(): Initializes the RippleAnimationWindow.
        update_animation(): Updates the animation.
        animate_point(index): Animates a point in the animation.
        create_zap_effect(circle): Creates a zap effect for a circle.
        expand_circle(circle): Expands a circle in the animation.
        remove_marked_circles(): Removes marked circles from the animation. (Garbage collection)
        move_centroids(): Moves the centroids based on the activated points.
    """

    def __init__(self):

        self.app = QApplication(sys.argv)
        self.central_widget = pg.Qt.QtWidgets.QWidget()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-10, 10)
        self.plot_widget.setXRange(-10, 10)
        self.plot_widget.setGeometry(0, 110, 1024, 768)
        # remove axis
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        self.plot_widget.show()

        self.data = np.zeros(N)
        self.centroid_positions = [
            (np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(N)
        ]
        self.indicators, self.indicator_traces = self.draw_centroids()

        # Add function that will be called when the timer times out
        self.frame = 0

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QApplication.instance().exec_()

    def animation(self):
        timer = QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)
        self.start()

    def update(self):
        """
        Updates the animation by removing marked circles, updating the binary vector,
        animating points, moving centroids, and stopping the timer when the animation is complete.
        """
        stime = time.perf_counter_ns()
        self.data = spike[0]
        spike[0] = np.zeros((50, 1))
        # Check and remove all marked circles (finished ripple animations)
        self.remove_marked_circles()

        # Update centroid location and replot centroid indicator
        self.move_centroids()
        self.update_centroid_indicator()

        # Update the binary vector and generate new spikes
        for i, val in enumerate(self.data):
            if val == 1:
                self.trigger_spike(i)
        self.frame += 1
        # print(f"FPS : {1 / (time.perf_counter_ns() - stime) * 1e9}")

    def trigger_spike(self, index):
        """
        Triggers a spike animation at the specified index.

        Args:
            index (int): The index of the spike to trigger.
        """

        pos = [self.centroid_positions[index]]

        # Add the ripple background glow element
        ripple_bg = pg.ScatterPlotItem()
        ripple_bg.setData(
            pos=pos,
            size=10,
            symbol="o",
            pen=pg.mkPen(width=8, color=(0, 255, 65, 100)),
            brush=(255, 0, 0, 0),
        )
        self.plot_widget.addItem(ripple_bg)
        QTimer.singleShot(0, lambda: self.expand_circle((ripple_bg, 0)))

        # Add the ripple element
        ripple = pg.ScatterPlotItem()
        ripple.setData(
            pos=pos,
            size=8,
            symbol="o",
            pen=pg.mkPen(width=3, color=(125, 255, 150, 255)),
            brush=(255, 0, 0, 0),
        )
        self.plot_widget.addItem(ripple)
        QTimer.singleShot(0, lambda: self.expand_circle((ripple, 0)))

        # Add the zapping effect element
        zap_effect = pg.ScatterPlotItem()
        zap_effect.setData(
            pos=pos,
            size=8,
            symbol="o",
            brush=(255, 230, 230, 255),
        )
        self.plot_widget.addItem(zap_effect)
        QTimer.singleShot(0, lambda: self.create_zap_effect(zap_effect))

    def create_zap_effect(self, circle):
        """
        Creates a zap effect for a circle by making it invisible after a delay.

        Args:
            circle (ScatterPlotItem): The circle to create the zap effect for.
        """
        QTimer.singleShot(30, lambda: circle.setPointsVisible(False))

    def expand_circle(self, circle_iter_tuple):
        """
        Expands a circle in the animation by gradually increasing its size.

        Args:
            circle (tuple): A tuple containing the circle item and its current radius.
        """
        circle, iter = circle_iter_tuple
        current_pen = circle.opts.get("pen")
        current_color = list(current_pen.color().getRgb())
        radius = circle.opts.get("size")

        if iter < 20 and current_color[-1] > 10 and radius < 40:
            circle.setSize(radius + 2)
            current_width = current_pen.width()
            current_color[-1] *= 0.85
            circle.setPen(
                pg.mkPen(
                    width=current_width,
                    color=current_color,
                )
            )
            QTimer.singleShot(1, lambda: self.expand_circle((circle, iter + 1)))
        else:
            circle.setPointsVisible(False)

    def remove_marked_circles(self):
        """
        Removes marked circles from the animation.
        """
        scatter_items = [
            item
            for item in self.plot_widget.items()
            if isinstance(item, pg.ScatterPlotItem)
        ]
        for item in scatter_items:
            if item.data.size > 0:
                if not all(item.data["visible"]):
                    self.plot_widget.removeItem(item)

    def draw_centroids(self):
        """
        Draws the centroid indicators in the animation.

        Returns:
            list: The centroid indicator items in the animation.
        """
        indicator = pg.ScatterPlotItem()
        indicator.setData(
            pos=self.centroid_positions,
            size=3,
            symbol="o",
            brush=(200, 200, 200, 200),
        )
        self.plot_widget.addItem(indicator)

        indicator_traces = []
        for i in range(N):
            indicator_trace = pg.PlotCurveItem()
            indicator_trace.setData(x=[0, 0], y=[0, 0])
            indicator_traces.append(indicator_trace)
            self.plot_widget.addItem(indicator_trace)
        return indicator, indicator_traces

    def update_centroid_indicator(self):
        """
        Updates the centroid indicator positions in the animation.
        """
        pos = self.indicators.data
        for i, trace in enumerate(self.indicator_traces):
            trace.setData(
                x=[pos[i][0], self.centroid_positions[i][0]],
                y=[pos[i][1], self.centroid_positions[i][1]],
            )

        self.indicators.setData(pos=self.centroid_positions, skipFiniteCheck=True)
        # pass

    def move_centroids(self):
        """
        Moves the centroids based on the activated points.
        """
        activated_positions = [
            (i, self.centroid_positions[i])
            for i, val in enumerate(self.data)
            if val == 1
        ]

        unactivated_positions = [
            (i, self.centroid_positions[i])
            for i, val in enumerate(self.data)
            if val == 0
        ]

        if len(activated_positions) > 1:
            second_elements = np.array([t[1] for t in activated_positions])
            # Calculating the mean vector of the second elements
            mean_vector = np.mean(second_elements, axis=0)
            centroid_of_activated = mean_vector

            for i, center in activated_positions:
                direction = centroid_of_activated - np.array(center)
                self.centroid_positions[i] += 0.1 * direction

        for i, center in activated_positions:
            repulsion_force = np.array([0.0, 0.0])
            for j, other_center in unactivated_positions:
                if i == j:
                    continue
                direction = np.array(other_center) - np.array(center)
                distance = np.linalg.norm(direction)
                if distance < 3:
                    repulsion_force -= direction / (distance**2)
            self.centroid_positions[i] += 0.5 * repulsion_force


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
    asyncio_thread = threading.Thread(target=asyncio_run)
    asyncio_thread.start()

    anim = RippleAnimation()
    anim.animation()
