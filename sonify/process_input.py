import cv2
import numpy as np
import time

# Set the desired width and height for the video capture
width = 320
height = 240

# Initialize video capture with the desired width and height
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Failed to open camera")
    exit()

# Read first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Failed to capture frame")
    exit()

# Resize the first frame
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Main loop
while True:
    start_t = time.perf_counter_ns()
    # Read current frame
    ret, curr_frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Resize current frame
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Extract horizontal component of flow
    flow_x = flow[..., 0]

    # Compute horizontal motion energy
    motion_energy = np.mean(flow_x)

    prev_gray = curr_gray.copy()
    print(
        f"Motion Energy: {motion_energy} and took {(time.perf_counter_ns() - start_t)/1e6} ms"
    )
    # Display current frame
    cv2.imshow("Motion Analysis", curr_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Update previous frame and grayscale image
    prev_gray = curr_gray.copy()

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
