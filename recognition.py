from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
import cv2
from calibration import Calibration
import pickle
import os
import time
import numpy as np
from math import sqrt
import sys


class Recognition:
    def __init__(self, q):
        self.queue = q

    def load_model(self):
        # device = select_device(device='mps', verbose=False)
        # model = YOLO('yolov8n.pt').to(device)
        model = YOLO('yolov8n.pt')
        return model

    def load_calibration(self):
        if os.path.isfile("calib/out.pickle"):
            try:
                with open("calib/out.pickle", "rb") as obj:
                    calib = pickle.load(obj)
                    return calib.get_params()
            except Exception as e:
                print("An error occurred when trying to load calibration data.")
                exit()
        else:
            calib = Calibration(calibrate=False)
            return calib.get_params()

    def init_recog(self):
        self.model = self.load_model()
        self.camera, self.distortion = self.load_calibration()
        self.camera = np.array([[764,0,662], [0,765, 325],[0,0,1]])
        self.distortion = np.array([0.0718, -0.084, -0.001, 0.001, 0.022])
        self.camera = np.array([[1056, 0.00000000, 665],
                        [0.00000000, 1057, 360],
                        [0.00000000, 0.00000000, 1.00000000]])

        self.distortion = np.array([0.0718, -0.084, 0.0012, 0.001, 0.022])

    def getAngle(self, realCoord):
        coord_x, coord_y, coord_z = realCoord
        theta_x = np.degrees(np.arctan(coord_x / coord_z))
        mag = sqrt(coord_x**2+coord_y**2+coord_z**2)
        theta_y = -np.degrees(np.arcsin(coord_y / mag))
        return theta_x, theta_y

    def point2angle(self, point, width, height, horizontal_fov, vertical_fov):
        x, y = point
        theta_x = (x - width / 2) / width * horizontal_fov
        theta_y = (height / 2 - y) / height * vertical_fov
        return theta_x, theta_y

    def start_process(self):
        self.init_recog()
        width = 1920
        height = 1080
        diagonal_fov = np.radians(66.0)
        diagonal_pixels = sqrt(width**2+height**2)
        horizontal_fov = np.degrees(2*np.arctan(np.tan(diagonal_fov/2)*width/diagonal_pixels))
        vertical_fov = np.degrees(2*np.arctan(np.tan(diagonal_fov/2)*height/diagonal_pixels))
        self.capture = cv2.VideoCapture(0)
        # use the yolo model to track relevant objects
        # person, bicycle, car, motorcycle, bus, train, truck, traffic light, stop sign, umbrella, cell phone, laptop. - 0, 1, 2, 3, 5, 6, 7, 9, 11, 25, 63, 67
        classes = {0: "person",1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck", 9: "traffic light", 11: "stop sign", 63: "cell phone", 67: "laptop",  39: "bottle", 66: "keyboard"}
        last_frame = time.time()
        num_frames = 0
        status, self.frame = self.capture.read()
        while self.capture.isOpened():
            cur_time = time.time()
            if cur_time - last_frame >= 1:
                sys.stderr.write(f"fps: {num_frames}\n")
                num_frames = 0
                last_frame = cur_time
            num_frames += 1
            status, self.frame = self.capture.read()
            # results = self.model.track(source=self.frame, classes=[0,2,9,11,39,66], device="mps", persist=True, conf = 0.05)
            results = self.model.track(source=self.frame, classes=[0, 1, 2, 3, 5, 6, 7, 9, 11, 25, 63, 67], persist=True, conf = 0.05)
            annotated_frame = results[0].plot()
            h, w = self.frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera, self.distortion, (w,h), 1, (w,h))
            # map ids to classes and angles
            angles = {} # map classes to angles
            # obtain the center point of each bounding box (for now)
            # account for objects detected without an id
            next_id = 100
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu()[0])
                centerX = (x1+x2) // 2
                centerY = (y1+y2) // 2

                # undistort the center point
                undistortedCenter = cv2.undistortPoints(np.array([[centerX, centerY]], dtype=np.float32), self.camera, self.distortion, None, newcameramtx)[0][0]
                undistortedLeft = cv2.undistortPoints(np.array([[x1, centerY]], dtype=np.float32), self.camera, self.distortion, None, newcameramtx)[0][0]
                undistortedRight = cv2.undistortPoints(np.array([[x2, centerY]], dtype=np.float32), self.camera, self.distortion, None, newcameramtx)[0][0]

                cv2.circle(annotated_frame, (centerX, centerY), radius=5, color=(0, 255, 0), thickness=-1)
                cv2.circle(annotated_frame, (x1, centerY), radius=5, color=(0, 255, 0), thickness=-1)
                cv2.circle(annotated_frame, (x2, centerY), radius=5, color=(0, 255, 0), thickness=-1)

                undistorted_theta_x, undistorted_theta_y = self.point2angle(undistortedCenter, width, height, horizontal_fov, vertical_fov)

                theta_left_x, theta_left_y = self.point2angle(undistortedLeft, width, height, horizontal_fov, vertical_fov)
                theta_right_x, theta_right_y = self.point2angle(undistortedRight, width, height, horizontal_fov, vertical_fov)

                if box.id != None:
                    angles[int(box.id)] = (undistorted_theta_x, int(box.cls))
                else:
                    angles[next_id] = (undistorted_theta_x, int(box.cls))
                    next_id += 1
                try:
                    print(f"class: {classes[int(box.cls)]} undistorted angle: {undistorted_theta_x}°, {undistorted_theta_y}°, {theta_left_x}°, {theta_left_y}°, {theta_right_x}°, {theta_right_y}°")
                except:
                    continue

            # empty the queue
            while not self.queue.empty():
                self.queue.get()

            # add the newest element(s) to the queue
            self.queue.put(angles)

            cv2.imshow("vid", annotated_frame)
            # cv2.setMouseCallback('vid', self.click_event)
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
            #TODO: check if this is okay
            # self.frame = self.next_frame
        cv2.destroyAllWindows()
        self.capture.release()
        cv2.waitKey(1)
