from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device, model_info
import cv2
from calibration import Calibration
import pickle
import os
import time
import numpy as np
from math import sqrt
import sys
from contextlib import redirect_stdout
from multiprocessing import Queue


class Recognition:
    # def __init__(self, conn):
    #     self.conn = conn
        # self.process()
    def __init__(self, q):
        self.queue = q

    def load_model(self):
        device = select_device(device='mps', verbose=False)
        model = YOLO('yolov8n.pt').to(device)
        return model

    def load_calibration(self):
        if os.path.isfile("calib/out.pickle"):
            try:
                with open("calib/out.pickle", "rb") as obj:
                    return pickle.load(obj)
            except Exception as e:
                print("An error occurred when trying to load calibration data.")
                exit()
        else:
            return Calibration(calibrate=False)

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.a = x
            self.b = y
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.c = x
            self.d = y


    def getAngle(self,realCoord):
        coord_x = realCoord[0]
        coord_y = realCoord[1]
        coord_z = realCoord[2]
        theta_x = np.degrees(np.arctan(coord_x / coord_z))
        mag = sqrt(coord_x**2+coord_y**2+coord_z**2)
        theta_y = -np.degrees(np.arcsin(coord_y / mag))
        return theta_x, theta_y

    def start_process(self):
        self.model = self.load_model()
        self.calib = self.load_calibration()
        self.calib.camera = np.array([[764,0,662], [0,765, 325],[0,0,1]])
        self.calib.distortion[0] = np.array([0.0718, -0.084, -0.001, 0.001, 0.022])
        self.a = 66
        self.b = 66
        self.c = 66
        self.d = 66
        width = 1920
        height = 1080
        # diagonal_fov = np.radians(66.0)
        # aspect_ratio = width / height
        # vertical_fov = 2 * np.arctan(np.tan(diagonal_fov / 2) / sqrt(1 + aspect_ratio**2))
        # horizontal_fov = 2 * np.arctan(aspect_ratio * np.tan(vertical_fov / 2));
        diagonal_fov = np.radians(66.0)
        totalcenter = [width/2,height/2]
        diagonal_pixels = sqrt(width**2+height**2)
        horizontal_fov = np.degrees(2*np.arctan(np.tan(diagonal_fov/2)*width/diagonal_pixels))
        vertical_fov = np.degrees(2*np.arctan(np.tan(diagonal_fov/2)*height/diagonal_pixels))
        pixelsToAngle = 66/2292.19
        camera = self.calib.cam
        distortion = self.calib.distortion
        capture = cv2.VideoCapture(0)
        # use the yolo model to track relevant objects
        classes = {0: "person", 2: "car", 9: "traffic light", 11: "stop sign", 39: "bottle", 66: "keyboard"}
        last_frame = time.time()
        num_frames = 0
        while capture.isOpened():
            cur_time = time.time()
            if cur_time - last_frame >= 1:
                sys.stderr.write(f"fps: {num_frames}\n")
                num_frames = 0
                last_frame = cur_time
            num_frames += 1
            status, frame = capture.read()
            # 0: person, 2: car, 9: traffic light, 11: stop sign, 39: bottle, 66: keyboard
            with redirect_stdout(open(os.devnull, 'w')):
                results = self.model.track(source=frame, classes=[0,2,9,11,39,66], device="mps", persist=True)
            annotated_frame = results[0].plot()
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera, distortion, (w,h), 1, (w,h))
            angles = {} # map ids to angles
            # obtain the center point of each bounding box (for now)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu()[0])
                centerX = (x1+x2) // 2
                centerY = (y1+y2) // 2
                # centerX = 1240
                # centerY = 490
                # centerX = self.a
                # centerY = self.b
                center = np.array([centerX , centerY, 1])
                # undistort the center point
                undistortedCenter = cv2.undistortPoints(np.array([[centerX, centerY]], dtype=np.float32), camera, distortion, None, newcameramtx)[0][0]
                cv2.circle(annotated_frame, (centerX, centerY), radius=5, color=(0, 255, 0), thickness=-1)
                cv2.circle(annotated_frame, (self.c, self.d), radius=5, color=(0, 0, 255), thickness=-1)
                centerX = undistortedCenter[0]
                centerY = undistortedCenter[1]
                undistorted_theta_x = (centerX - width / 2) / width * horizontal_fov*1.25
                undistorted_theta_y = (height / 2 - centerY) / height * vertical_fov
                angles[int(box.id)] = undistorted_theta_x
                # print(undistorted_theta_x)
                # self.conn.send(undistorted_theta_x.item())
                try:
                    # pass
                    print(f"class: {classes[int(box.cls)]} id: {int(box.id)} undistorted angle: {undistorted_theta_x}°, {undistorted_theta_y}°")
                    # print(66.0/horizontal_fov)
                    # print(distortion)
                except:
                    continue
                            # empty the queue
            while not self.queue.empty():
                self.queue.get()
            # add the newest element(s) to the queue
            self.queue.put(angles)
            cv2.imshow("vid", annotated_frame)
            cv2.setMouseCallback('vid', self.click_event)
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        capture.release()
        cv2.waitKey(1)
