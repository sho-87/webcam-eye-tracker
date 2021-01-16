import cv2
import time
import d3dshot
import numpy as np

from datetime import datetime
from collections import deque
from Gaze import Detector, Predictor
from Models import FullModel
from utils import get_config, clamp_value


# Read config.ini file
SETTINGS, COLOURS, EYETRACKER, TF = get_config("config.ini")

# Load trained model
detector = Detector(output_size=SETTINGS["image_size"])
predictor = Predictor(
    FullModel,
    model_data="trained_models/eyetracking_model.pt",
    config_file="trained_models/eyetracking_config.json",
)
screen_errors = region_map = np.load("trained_models/eyetracking_errors.npy")

track_x = deque(
    [0] * SETTINGS["avg_window_length"], maxlen=SETTINGS["avg_window_length"]
)
track_y = deque(
    [0] * SETTINGS["avg_window_length"], maxlen=SETTINGS["avg_window_length"]
)
track_error = deque(
    [0] * (SETTINGS["avg_window_length"] * 2),
    maxlen=SETTINGS["avg_window_length"] * 2,
)

w = 1920
h = 1080
videoWriter = None
if EYETRACKER["write_to_disk"]:
    # FIXME: fix video codec on windows
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    videoWriter = cv2.VideoWriter(
        "media/recordings/{}.mp4".format(date_time),
        fourcc,
        EYETRACKER["tracker_frame_rate"],
        (
            int(w * EYETRACKER["screen_capture_scale"]),
            int(h * EYETRACKER["screen_capture_scale"]),
        ),
    )

d = d3dshot.create(capture_output="numpy")
d.display = d.displays[0]
d.capture()

last_time = time.time()
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # wait for escape key
        detector.close()
        cv2.destroyAllWindows()
        d.stop()
        break

    cur_time = time.time()

    if (cur_time - last_time) >= 1 / 60:
        fps = 1 / (cur_time - last_time)
        last_time = cur_time

        # Get camera data
        l_eye, r_eye, face, face_align, head_pos, angle = detector.get_frame()

        # Get screenshot
        screenshot = d.get_latest_frame().copy()

        # Overlays
        x_hat, y_hat = predictor.predict(
            face, l_eye, r_eye, head_pos, head_angle=angle
        )

        track_x.append(x_hat)
        track_y.append(y_hat)

        x_hat_clamp = clamp_value(x_hat, w)
        y_hat_clamp = clamp_value(y_hat, h)
        error = screen_errors[int(x_hat_clamp) - 1][int(y_hat_clamp) - 1]
        track_error.append(error * 0.75)

        weights = np.arange(1, SETTINGS["avg_window_length"] + 1)
        weights_error = np.arange(1, (SETTINGS["avg_window_length"] * 2) + 1)

        cv2.circle(
            screenshot,
            (
                int(np.average(track_x, weights=weights)),
                int(np.average(track_y, weights=weights)),
            ),
            int(np.average(track_error, weights=weights_error)),
            (255, 255, 255, 50),
            -1,
        )

        cv2.circle(
            screenshot,
            (
                int(np.average(track_x, weights=weights)),
                int(np.average(track_y, weights=weights)),
            ),
            int(np.average(track_error, weights=weights_error)),
            COLOURS["green"],
            5,
        )

        cv2.putText(
            screenshot,
            "fps: {}".format(round(fps, 2)),
            (0, h),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            COLOURS["green"],
        )

        if EYETRACKER["show_webcam"]:
            large_size = SETTINGS["image_size"] * 2

            screenshot[0:large_size, 0:large_size, 0:3] = cv2.resize(
                face_align, (large_size, large_size)
            )

            screenshot[
                0:large_size,
                large_size : large_size * 2,
                0:3,
            ] = cv2.resize(
                np.repeat(head_pos[:, :, np.newaxis], 3, axis=2),
                (large_size, large_size),
            )

            screenshot[
                large_size : large_size * 2,
                0:large_size,
                0:3,
            ] = cv2.resize(l_eye, (large_size, large_size))

            screenshot[
                large_size : large_size * 2,
                large_size : large_size * 2,
                0:3,
            ] = cv2.resize(r_eye, (large_size, large_size))

        # Resize and write frame
        screenshot = cv2.resize(
            screenshot,
            (
                int(w * EYETRACKER["screen_capture_scale"]),
                int(h * EYETRACKER["screen_capture_scale"]),
            ),
        )

        cv2.imshow("Eyetracker", screenshot)
        # cv2.imshow("Webcam", face)

        if EYETRACKER["write_to_disk"]:
            videoWriter.write(screenshot)

if EYETRACKER["write_to_disk"]:
    videoWriter.release()