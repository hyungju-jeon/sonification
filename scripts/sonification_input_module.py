import cv2
import numpy as np
import time
import asyncio
from pythonosc.udp_client import SimpleUDPClient
from sonification_communication_module import *

# ---------------------------------------------------------------- #
# Video capture related parameters
FRAME_WIDTH = 40
FRAME_HEIGHT = 30


class MotionEnergy:
    def __init__(self, verbose=False):
        self.FRAME = cv2.VideoCapture(0)
        self.FRAME.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.FRAME.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Check if the camera opened successfully
        if not self.FRAME.isOpened():
            print("Error: Failed to open camera")
            exit()
        # Read first frame
        ret, prev_frame = self.FRAME.read()
        if not ret:
            print("Error: Failed to capture frame")
            exit()
        self.prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        self.verbose = verbose
        self.OSCsender = SimpleUDPClient(SERVER_IP, MOTION_ENERGY_PORT)

    async def start(self):
        while True:
            start_time = time.perf_counter_ns()
            # Read current frame
            ret, curr_frame = self.FRAME.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

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

            if self.verbose:
                print(
                    f"Motion Energy X: {motion_energy_x}, Motion Energy Y: {motion_energy_y} took {(time.perf_counter_ns() - start_time)/1e6} ms"
                )
            self.display_text(curr_frame, motion_energy_x, motion_energy_y)

            # Display current frame
            cv2.imshow("Motion Analysis", curr_frame)

            # Update previous frame and grayscale image
            self.prev_gray = curr_gray.copy()

            self.send_packet(motion_energy_x, motion_energy_y)
            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.FRAME.release()
                cv2.destroyAllWindows()
                break
            await asyncio.sleep(0)

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

        if self.verbose:
            print(f"Sending packet: /MOTION_ENERGY {[x, y]}")


if __name__ == "__main__":
    motion_energy = MotionEnergy(True)
    asyncio.run(motion_energy.start())
    print("Motion Energy Analysis Complete")
    print("Exiting...")
