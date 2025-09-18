import cv2
import numpy as np

# --- Global Variables ---
# List to store the points that we want to track
points_to_track = []
# Flag to check if we have points selected
points_selected = False
# Previous frame, required for optical flow
old_gray = None
# Parameters for the Lucas-Kanade optical flow algorithm
# These values can be tweaked for performance vs. accuracy
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# --- Mouse Callback Function ---
def select_point(event, x, y, flags, param):
    """
    This function is called whenever a mouse event occurs in the window.
    It adds the coordinates of a left-click to our list of points to track.
    """
    global points_to_track, points_selected
    # If the event is a left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the new point to our list
        points_to_track.append((x, y))
        points_selected = True

# --- Main Application Logic ---
def main():
    """
    Initializes the camera, handles the main loop for video processing,
    and manages user input.
    """
    global points_to_track, points_selected, old_gray

    # Open a connection to the default camera (webcam)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Create a named window and set the mouse callback function
    window_name = 'Breathing Tracker - Click to add points, Press C to clear, Press Q to quit'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_point)

    # Convert the points list to a NumPy array for OpenCV functions
    p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)

    # Create a mask image for drawing the tracking lines
    # It will have the same dimensions and type as the video frames
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video stream.")
        cap.release()
        return
    mask = np.zeros_like(frame)

    while True:
        # Read a new frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Finished reading stream or error occurred.")
            break

        # Convert the frame to grayscale for the optical flow algorithm
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If points have been selected by the user
        if points_selected and len(points_to_track) > 0:
            # Convert the list of points to a NumPy array for the first time
            p0 = np.array(points_to_track, dtype=np.float32).reshape(-1, 1, 2)
            # Reset the list and the flag so we don't re-add points
            points_to_track = []
            # Initialize old_gray with the current frame
            old_gray = frame_gray.copy()

        # If we have points to track (p0 is not empty)
        if p0.any():
            # Calculate optical flow using Lucas-Kanade method
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select only the good points (where status 'st' is 1)
            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw the tracking lines and circles
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    # Draw a line on the mask from the old point to the new point
                    mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                    # Draw a circle at the new position of the point on the frame
                    frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

                # Add the drawings from the mask onto the original frame
                img = cv2.add(frame, mask)
                cv2.imshow(window_name, img)

                # Update the previous frame and previous points for the next iteration
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            else:
                # If tracking is lost, show the original frame
                cv2.imshow(window_name, frame)
        else:
            # If no points are being tracked yet, just show the frame
            cv2.imshow(window_name, frame)

        # Wait for a key press
        k = cv2.waitKey(30) & 0xff
        # If 'q' is pressed, break the loop
        if k == ord('q'):
            break
        # If 'c' is pressed, clear the points and the mask
        elif k == ord('c'):
            p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)
            mask = np.zeros_like(frame)
            points_to_track = []
            points_selected = False


    # Release the camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
