#multithreaded version
from multiprocessing import Process, Queue, freeze_support
from math import sqrt
import cv2
import numpy as np
import time
import sys

class MultiVideo:
    def __init__(self, model):

    def worker(task_queue):
        while True:
            task = task_queue.get()
            if task is None:
                break
            func, arg = task
            func(arg)

    def stopProcesses():
        for i in range(num_processes):
            task_queue.put(None)

    def getAngle(realCoord):
        coord_x = realCoord[0]
        coord_y = realCoord[1]
        coord_z = realCoord[2]
        theta_x = np.degrees(np.arctan(coord_x / coord_z))
        mag = sqrt(coord_x**2+coord_y**2+coord_z**2)
        theta_y = -np.degrees(np.arcsin(coord_y / mag))
        return theta_x, theta_y

    def readFrame(task_queue):
        status, frame = capture.read()
        task_queue.put((readFrame, None))
        task_queue.put((processFrame, frame))

    # use the yolo model to track relevant objects
    classes = {0: "person", 2: "car", 9: "traffic light", 11: "stop sign", 39: "bottle", 66: "keyboard"}
    last_frame = time.time()
    num_frames = 0
    while capture.isOpened():
        cur_time = time.time()
        if cur_time - last_frame >= 1:
            # clear_output(wait=True)
            clear_output()
            sys.stderr.write(f"fps: {num_frames}\n")
            num_frames = 0
            last_frame = cur_time
        num_frames += 1
        status, frame = capture.read()
        # 0: person, 2: car, 9: traffic light, 11: stop sign, 39: bottle, 66: keyboard
        results = model.track(source=frame, classes=[0,2,9,11,39,66], device="mps", persist=True)
        annotated_frame = results[0].plot()
        #testing stuff rn
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera, distortion, (w,h), 1, (w,h))

        # obtain the center point of each bounding box (for now)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu()[0])
            centerX = (x1+x2) // 2
            centerY = (y1+y2) // 2
            center = np.array([centerX , centerY, 1])
            # undistort the center point
            undistortedCenter = cv2.undistortPoints(np.array([[centerX, centerY]], dtype=np.float32), camera, distortion, None, newcameramtx)
            cv2.circle(annotated_frame, (centerX, centerY), radius=5, color=(0, 255, 0), thickness=-1)
            #TODO: determine whether determining undistorted world coord or distorted one is more accurate
            realCoord = np.linalg.solve(camera, center)
            undistortedCoord = np.linalg.solve(newcameramtx, np.array([undistortedCenter[0][0][0], undistortedCenter[0][0][1], 1]))
            # distorted_theta_x, distorted_theta_y = getAngle(realCoord) 
            undistorted_theta_x, undistorted_theta_y = getAngle(undistortedCoord)
            try:
                print(f"class: {classes[int(box.cls)]} id: {int(box.id)} undistorted angle: {undistorted_theta_x}°, {undistorted_theta_y}°")
            except:
                continue
        cv2.imshow("vid", annotated_frame)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    capture.release()
    cv2.waitKey(1)

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    task_queue = Queue()
    num_processes = 2
    processes = []

    for i in range(num_processes):
        p = Process(target=worker,args=(task_queue,))
        p.start()
        processes.append(p)
    
    task_queue.put((readFrame, task_queue))

    for p in processes:
        p.join()
    cv2.destroyAllWindows()
    capture.release()
    cv2.waitKey(1)
