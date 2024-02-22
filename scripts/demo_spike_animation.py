import asyncio
import sys
import threading
import time
import numpy as np
import pyqtgraph as pg


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
    """
    A QMainWindow subclass for displaying a ripple animation.

    Attributes:
        central_widget (QWidget): The central widget of the main window.
        plot_widget (PlotWidget): The plot widget for displaying the animation.
        layout (QVBoxLayout): The layout for arranging the widgets.
        data (ndarray): The binary spike vector used for animation.
        centroid_positions (list): The positions of the spike centroids.
        frame (int): The current frame of the animation.
        timer (QTimer): The timer for updating the animation.
        indicators (list): The centroid indicator items in the animation.

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
        super().__init__()
        self.setWindowTitle("Ripple Animation")
        self.central_widget = pg.Qt.QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-10, 10)
        self.plot_widget.setXRange(-10, 10)
        self.layout = pg.Qt.QtWidgets.QVBoxLayout(self.central_widget)
        # vb = plt.getViewBox()
        # vb.setAspectLocked(lock=False)
        # vb.enableAutoRange(enable=False)

        # remove axis
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")

        self.layout.addWidget(self.plot_widget)
        self.data = np.zeros(N)
        self.centroid_positions = [
            (np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(N)
        ]
        self.indicators = self.draw_centroids()

        # Add function that will be called when the timer times out
        self.frame = 0
        self.timer = QTimer()

        self.timer.timeout.connect(self.update_animation)
        self.timer.start(10)  # Adjust the refresh rate (in ms) as needed

    def update_animation(self):
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
        print(f"FPS : {1 / (time.perf_counter_ns() - stime) * 1e9}")

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
            size=5,
            symbol="o",
            pen=pg.mkPen(width=8, color=(0, 255, 65, 200)),
            brush=(255, 0, 0, 0),
        )
        self.plot_widget.addItem(ripple_bg)
        QTimer.singleShot(1, lambda: self.expand_circle((ripple_bg, 0)))

        # Add the ripple element
        ripple = pg.ScatterPlotItem()
        ripple.setData(
            pos=pos,
            size=5,
            symbol="o",
            pen=pg.mkPen(width=3, color=(150, 255, 150)),
            brush=(255, 0, 0, 0),
        )
        self.plot_widget.addItem(ripple)
        QTimer.singleShot(1, lambda: self.expand_circle((ripple, 0)))

        # Add the zapping effect element
        zap_effect = pg.ScatterPlotItem()
        zap_effect.setData(
            pos=pos,
            size=5,
            symbol="o",
            brush=(255, 230, 230, 255),
        )
        self.plot_widget.addItem(zap_effect)
        QTimer.singleShot(1, lambda: self.create_zap_effect(zap_effect))

    def create_zap_effect(self, circle):
        """
        Creates a zap effect for a circle by making it invisible after a delay.

        Args:
            circle (ScatterPlotItem): The circle to create the zap effect for.
        """
        QTimer.singleShot(100, lambda: circle.setPointsVisible(False))

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

        if iter < 20 and current_color[-1] > 10 and radius < 30:
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
            size=1,
            symbol="o",
            brush=(200, 200, 200, 200),
        )
        self.plot_widget.addItem(indicator)

        return indicator

    def update_centroid_indicator(self):
        """
        Updates the centroid indicator positions in the animation.
        """
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

        for i, center in enumerate(self.centroid_positions):
            repulsion_force = np.array([0.0, 0.0])
            for j, other_center in unactivated_positions:
                if i == j:
                    continue
                direction = np.array(other_center) - np.array(center)
                distance = np.linalg.norm(direction)
                if distance < 1:
                    repulsion_force -= direction / (distance**2)
            self.centroid_positions[i] += (
                0.1 * repulsion_force / len(unactivated_positions)
            )


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

    app = QApplication(sys.argv)
    window = RippleAnimationWindow()
    window.show()

    sys.exit(app.exec_())
