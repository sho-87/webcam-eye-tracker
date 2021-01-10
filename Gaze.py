import os
import cv2
import dlib
import json
import queue
import threading
import torch
import numpy as np

from collections import OrderedDict
from torchvision import transforms
from utils import get_config, shape_to_np

# Read config.ini file
SETTINGS, COLOURS, EYETRACKER, TF = get_config("config.ini")


class Detector:
    def __init__(
        self,
        output_size,
        show_stream=False,
        show_markers=False,
        show_output=False,
        gpu=1
    ):
        print("Starting face detector...")
        self.output_size = output_size
        self.show_stream = show_stream
        self.show_output = show_output
        self.show_markers = show_markers
        self.face_img = np.zeros((output_size, output_size, 3))
        self.face_align_img = np.zeros((output_size, output_size, 3))
        self.l_eye_img = np.zeros((output_size, output_size, 3))
        self.r_eye_img = np.zeros((output_size, output_size, 3))
        self.head_pos = np.ones((output_size, output_size))
        self.head_angle = 0.0

        # Models for face detection and landmark prediction
        dlib.cuda.set_device(gpu)
        self.landmark_idx = OrderedDict([("right_eye", (0, 2)), ("left_eye", (2, 4))])
        self.detector = dlib.cnn_face_detection_model_v1(
            "trained_models/mmod_human_face_detector.dat"
        )
        self.predictor = dlib.shape_predictor(
            "trained_models/shape_predictor_5_face_landmarks.dat"
        )

        # Threaded webcam capture
        self.capture = cv2.VideoCapture(0)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def get_frame(self):
        frame = self.q.get()
        dets = self.detector(frame, 0)

        if len(dets) == 1:
            # Get feature locations
            features = self.predictor(frame, dets[0].rect)
            reshaped = shape_to_np(features)

            l_start, l_end = self.landmark_idx["left_eye"]
            r_start, r_end = self.landmark_idx["right_eye"]
            l_eye_pts = reshaped[l_start:l_end]
            r_eye_pts = reshaped[r_start:r_end]
            l_eye_width = l_eye_pts[1][0] - l_eye_pts[0][0]
            r_eye_width = r_eye_pts[0][0] - r_eye_pts[1][0]

            # Calculate eye centers and head angle
            l_eye_center = l_eye_pts.mean(axis=0).astype("int")
            r_eye_center = r_eye_pts.mean(axis=0).astype("int")
            eye_dist = np.linalg.norm(r_eye_center - l_eye_center)
            dY = r_eye_center[1] - l_eye_center[1]
            dX = r_eye_center[0] - l_eye_center[0]
            self.head_angle = np.degrees(np.arctan2(dY, dX))

            if self.show_markers:
                for point in l_eye_pts:
                    cv2.circle(frame, (point[0], point[1]), 1, COLOURS["blue"], -1)

                for point in r_eye_pts:
                    cv2.circle(frame, (point[0], point[1]), 1, COLOURS["blue"], -1)

                cv2.circle(
                    frame, (l_eye_center[0], l_eye_center[1]), 3, COLOURS["green"], 1
                )
                cv2.circle(
                    frame, (r_eye_center[0], r_eye_center[1]), 3, COLOURS["green"], 1
                )

            # Face extraction and alignment
            desired_l_eye_pos = (0.35, 0.5)
            desired_r_eye_posx = 1.0 - desired_l_eye_pos[0]

            desired_dist = desired_r_eye_posx - desired_l_eye_pos[0]
            desired_dist *= self.output_size
            scale = desired_dist / eye_dist

            eyes_center = (
                (l_eye_center[0] + r_eye_center[0]) // 2,
                (l_eye_center[1] + r_eye_center[1]) // 2,
            )

            t_x = self.output_size * 0.5
            t_y = self.output_size * desired_l_eye_pos[1]

            align_angles = (0, self.head_angle)
            for angle in align_angles:
                M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
                M[0, 2] += t_x - eyes_center[0]
                M[1, 2] += t_y - eyes_center[1]

                aligned = cv2.warpAffine(
                    frame,
                    M,
                    (self.output_size, self.output_size),
                    flags=cv2.INTER_CUBIC,
                )

                if angle == 0:
                    self.face_img = aligned
                else:
                    self.face_align_img = aligned

            # Get eyes (square regions based on eye width)
            try:
                l_eye_img = frame[
                    l_eye_center[1]
                    - int(l_eye_width / 2) : l_eye_center[1]
                    + int(l_eye_width / 2),
                    l_eye_pts[0][0] : l_eye_pts[1][0],
                ]
                self.l_eye_img = cv2.resize(
                    l_eye_img, (self.output_size, self.output_size)
                )

                r_eye_img = frame[
                    r_eye_center[1]
                    - int(r_eye_width / 2) : r_eye_center[1]
                    + int(r_eye_width / 2),
                    r_eye_pts[1][0] : r_eye_pts[0][0],
                ]
                self.r_eye_img = cv2.resize(
                    r_eye_img, (self.output_size, self.output_size)
                )
            except:
                pass

            # Get position of head in the frame
            frame_bw = np.ones((frame.shape[0], frame.shape[1])) * 255
            cv2.rectangle(
                frame_bw,
                (dets[0].rect.left(), dets[0].rect.top()),
                (dets[0].rect.right(), dets[0].rect.bottom()),
                COLOURS["black"],
                -1,
            )
            self.head_pos = cv2.resize(frame_bw, (self.output_size, self.output_size))

            if self.show_output:
                cv2.imshow("Head position", self.head_pos)
                cv2.imshow(
                    "Face and eyes",
                    np.vstack(
                        (
                            np.hstack((self.face_img, self.face_align_img)),
                            np.hstack((self.l_eye_img, self.r_eye_img)),
                        )
                    ),
                )

            if self.show_stream:
                cv2.imshow("Webcam", frame)

        return (
            self.l_eye_img,
            self.r_eye_img,
            self.face_img,
            self.face_align_img,
            self.head_pos,
            self.head_angle,
        )

    def close(self):
        print("Closing face detector...")
        self.capture.release()
        cv2.destroyAllWindows()


class Predictor:
    def __init__(self, model, model_data, config_file=None, gpu=1):
        super().__init__()

        _, ext = os.path.splitext(model_data)
        if ext == ".ckpt":
            self.model = model.load_from_checkpoint(model_data)
        else:
            with open(config_file) as json_file:
                config = json.load(json_file)
            self.model = model(config)
            self.model.load_state_dict(torch.load(model_data))

        self.gpu = gpu
        self.model.double()
        self.model.cuda(self.gpu)
        self.model.eval()

    def predict(self, *img_list, head_angle=None):
        images = []
        for img in img_list:
            if not img.dtype == np.uint8:
                img = img.astype(np.uint8)
            img = transforms.ToTensor()(img).unsqueeze(0)
            img = img.double()
            img = img.cuda(self.gpu)
            images.append(img)

        if head_angle is not None:
            angle = torch.tensor(head_angle).double().flatten().cuda(self.gpu)
            images.append(angle)

        with torch.no_grad():
            coords = self.model(*images)
            coords = coords.cpu().numpy()[0]

        return coords[0], coords[1]


if __name__ == "__main__":
    detector = Detector(
        output_size=512, show_stream=True, show_output=True, show_markers=False
    )

    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # wait for escape key
            break
        detector.get_frame()

    detector.close()