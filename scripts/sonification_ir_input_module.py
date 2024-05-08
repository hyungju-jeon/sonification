import cv2
import numpy as np
import datetime as dt
import time
import asyncio
import sys

import matplotlib.pyplot as plt

from PIL import Image
from pseyepy import Camera, Display, Stream

from utils import plotting

#d = Display(c) # begin the display

from pythonosc.udp_client import SimpleUDPClient
from scripts.check_UDP_latency import elapsed_time_update
from sonification_communication_module import *

# ---------------------------------------------------------------- #
# Video capture related parameters
FRAME_WIDTH = 120
FRAME_HEIGHT = 90

# Optical Flow Config
OF_LOGIT_TRANSFOEM = True
NORMALIZE = True
OF_CLIP = True

'''
After getting a recording sample for as much as possible movement variations we can set
min and max values for the optical flow those can be used for normalization.
'''
max_optical_flow = 1
max_optical_flow = - max_optical_flow
of_lower_cutoff = (np.abs(max_optical_flow) + max_optical_flow) / 0.1

x_scale = 1
y_scale = 1

# Optical Flow Logit Transormation Paramerters
weight = 1
bias = max_optical_flow / 2

# Frame Processing Config
FRAME_LOGIT_TRANSFORM = True
SEGMENT = True
NORMALIZE = True

frame_lower_cutoff = 50


class IRVideoCapture(Camera):
    '''this is a subclass'''

class MotionEnergy:

    def __init__(self, verbose=False):

        self.x_buffer_size = 60
        self.x_buffer = np.zeros(self.x_buffer_size)

        self.y_buffer_size = 60
        self.y_buffer = np.zeros(self.y_buffer_size)

        self.x_current_index = 0
        self.y_current_index = 0

        self.max_optical_flow = 0

        self.fig, self.ax = plt.subplots()
        plt.title('Optical Flow over time')
        plt.xlabel('Frame Window')
        plt.ylabel('Optical Flow')

        self.x_motion_line, = self.ax.plot(self.x_buffer)
        self.x_motion_line.set_label('Optical Flow X')

        self.y_motion_line, = self.ax.plot(self.y_buffer)
        self.y_motion_line.set_label('Optical Flow Y')

        if self.x_buffer_size > self.y_buffer_size:
            self.ax.set_xlim(0, self.x_buffer_size - 1)
        else:
            self.ax.set_xlim(0, self.y_buffer_size - 1)

        # Enable interactive mode
        plt.ion()

        try:
            # 63 is the max possible gain.
            self.FRAME = IRVideoCapture(0, gain=63)
        except:
            print("Error: Failed to open camera")
            exit()

        # Read first frame
        try:
            frame, timestamp = self.FRAME.read()
        except:
            print("Error: Failed to capture frame")
            exit()

        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.verbose = verbose
        self.OSCsender = SimpleUDPClient(LOCAL_SERVER, MOTION_ENERGY_PORT)
        self.InferenceOSCsender = SimpleUDPClient(LOCAL_SERVER, SPIKE_INFERENCE_PORT)


    async def start(self):

        while True:
            start_t = time.perf_counter_ns()

            # Read current frame
            try:
                curr_frame, timestamp = self.FRAME.read()
            except:
                print("Error: Failed to capture frame")
                break


            #print(curr_frame)
            #curr_frame = self.frame_transformation(curr_frame)

            # Resize current frame
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Compute optical flow using Lucas-Kanade method
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, curr_gray, None, 0.25, 2, 20, 1, 5, 1.1, 0
            )

            # Extract horizontal/vertical component of flow
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

            # Compute horizontal motion energy
            motion_energy_x = np.mean(flow_x)
            motion_energy_y = np.mean(flow_y)

            #motion_energy_x = (
            #    np.clip(motion_energy_x * 2 * np.pi, -np.pi, np.pi) * x_scale
            #)
            #motion_energy_y = np.clip(motion_energy_y, -1, 1) * y_scale

            if self.verbose:
                print(
                    f"Motion Energy X: {motion_energy_x}, Motion Energy Y: {motion_energy_y} took {(time.perf_counter_ns() - start_t)/1e6} ms"
                )

            #motion_energy_x, motion_energy_y = self.optical_flow_transformation(motion_energy_x, motion_energy_y)

            # Display current frame
            cv2.imshow("Motion Analysis", curr_frame)
            self.display_text(curr_frame, motion_energy_x, motion_energy_y)

            # Plot optical flow evolution
            self.plot(motion_energy_x, motion_energy_y)
            plt.show()

            # Update previous frame and grayscale image
            self.prev_gray = curr_gray.copy()

            self.send_packet(motion_energy_x, motion_energy_y)
            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.FRAME.release()
                cv2.destroyAllWindows()
                break

            elapsed_time = time.perf_counter_ns() - start_t
            sleep_duration = np.fmax(5e-2 * 1e9 - (time.perf_counter_ns() - start_t), 0)

            # if sleep_duration == 0 & self.verbose:
            #     print(
            #         f"Input iteration took {elapsed_time/1e6}ms which is longer than {5e-2*1e3} ms"
            #     )
            await busy_timer(sleep_duration)

        # Release video capture and close windows


    def display_text(self, curr_frame, motion_energy_x, motion_energy_y):
        # Display text whether moving to the right or left
        text_args = {
            "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
            "fontScale": 1,
            "color": (0, 0, 255),
            "thickness": 2,
        }
        if motion_energy_x > 0.3:
            cv2.putText(curr_frame, "Moving to the right", (10, 30), **text_args)
        elif motion_energy_x < -0.3:
            cv2.putText(curr_frame, "Moving to the left", (10, 30), **text_args)
        else:
            cv2.putText(curr_frame, "Not moving", (10, 30), **text_args)
        # Display text whether moving up or down
        if motion_energy_y > 0.3:
            cv2.putText(curr_frame, "Moving Down", (10, 60), **text_args)
        elif motion_energy_y < -0.3:
            cv2.putText(curr_frame, "Moving Up", (10, 60), **text_args)
        else:
            cv2.putText(curr_frame, "Not moving", (10, 60), **text_args)


    def send_packet(self, x, y):
        self.OSCsender.send_message(
            "/MOTION_ENERGY",
            [x.astype(float), y.astype(float)],
        )
        self.InferenceOSCsender.send_message(
            "/MOTION_ENERGY",
            [x.astype(float), y.astype(float)],
        )

        if self.verbose:
            print(f"Sending packet: /MOTION_ENERGY {[x, y]}")


    def update_x_buffer(self, motion_energy_x, index):
        self.x_buffer[index] = motion_energy_x

        if np.abs(motion_energy_x) > self.max_optical_flow:
            self.max_optical_flow = np.abs(motion_energy_x)

        self.ax.set_ylim(- self.max_optical_flow, self.max_optical_flow)

        return (index + 1) % self.x_buffer_size
    

    def update_y_buffer(self, motion_energy_y, index):
        self.y_buffer[index] = motion_energy_y

        if np.abs(motion_energy_y) > self.max_optical_flow:
            self.max_optical_flow = np.abs(motion_energy_y)

        self.ax.set_ylim(- self.max_optical_flow, self.max_optical_flow)

        return (index + 1) % self.y_buffer_size


    # Function to update the plot line with the current buffer
    def update_plot(self):

        self.x_motion_line.set_ydata(self.x_buffer)
    
        self.y_motion_line.set_ydata(self.y_buffer)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def plot(self, motion_energy_x, motion_energy_y):

        # Update each buffer with the new motion energy value
        self.x_current_index = self.update_x_buffer(motion_energy_x, self.x_current_index)
        self.y_current_index = self.update_y_buffer(motion_energy_y, self.y_current_index)
        
        # Update the plot with the current buffer data
        self.update_plot()


    def logit_transform(self, optical_flow):

        optical_flow = np.log((weight * (optical_flow + bias)) / (1 - (weight * (optical_flow + bias))))

        return optical_flow
    

    def clip(self, motion_energy):

        motion_energy = 0 if motion_energy < of_lower_cutoff else motion_energy

        return motion_energy


    def normalize_frame(frame):
        pass


    def segment_frame(self, frame):
        '''
        Make all pixels of the frame where the mean of the R,G,B channels
        is less than 50 into black (i.e 0)
        '''
        frame = frame[np.mean(frame, axis=-1) < frame_lower_cutoff] = 0

        return frame


    def increase_contrast(frame):

        frame[frame * 1.5 > 255] = 255


    def optical_flow_transformation(self, motion_energy_x, motion_energy_y):

        if OF_LOGIT_TRANSFOEM:
            motion_energy_x = self.logit_transform(motion_energy_x)
            motion_energy_y = self.logit_transform(motion_energy_y)

        if OF_CLIP:
            motion_energy_x = self.clip(motion_energy_x)
            motion_energy_y = self.clip(motion_energy_y)

        return motion_energy_x, motion_energy_y
    

    def frame_transformation(self, frame):

        if SEGMENT:
            frame = self.segment_frame(frame)

        return frame


if __name__ == "__main__":
    motion_energy = MotionEnergy(verbose=True)
    asyncio.run(motion_energy.start())
    print("Motion Energy Analysis Complete")
    print("Exiting...")
