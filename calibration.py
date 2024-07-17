import numpy as np
import cv2
import glob
import pickle


class Calibration:

    def __init__(self, calibrate: bool):
        self.status = None
        self.cam = None
        self.distortion = None
        self.r = None
        self.t = None
        if calibrate:
            self.calibrate_photos()
        self.calibrate()

    def calibrate_photos(self):
        # delete any calibration photos already present previously
        origPhotos = glob.glob("calib/*")
        for photo in origPhotos:
            os.remove(photo)

        # take a series of 10 photos
        capture = cv2.VideoCapture(0)
        numPhotos = 0
        while capture.isOpened():
            status, frame = capture.read()
            key = cv2.waitKey(25)
            cv2.imshow("vid", frame)
            if key == ord('s'):
                numPhotos += 1
                imgPath = f"calib/calib{numPhotos}.jpg"
                cv2.imwrite(imgPath, frame)
                print(f"saved to {imgPath}!")
            if numPhotos == 10 or key == ord('q'):
                break
        cv2.destroyAllWindows()
        capture.release()

    def calibrate(self):
        # calibration setup (assume we use a chessboard for calibration)
        # dimensions of chessboard: 6x8
        # NOTE: as we are solely using this calibration for determination of angle, the size of each chess square does not matter
        images = glob.glob("calib/*")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        objp = np.zeros((6*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)
        objpoints = []
        imgpoints = []
        for imgname in images:
            img = cv2.imread(imgname)
            # convert the image to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # identify corners of chessboard
            corner_flags = cv2.CALIB_CB_EXHAUSTIVE
            found, corners = cv2.findChessboardCornersSB(img_gray, (6,8), corner_flags)
            # skip the image if the chessboard corners were not properly found
            if not found:
                continue
            corners_refined = cv2.cornerSubPix(img_gray, corners, (4,4), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            cv2.drawChessboardCorners(img_gray, (6,8), corners_refined, found)
            cv2.imshow("calib", img_gray)
            cv2.waitKey(0)
        # calibrate the camera
        status, self.cam, self.distortion, self.r, self.t = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)
        try:
            with open("calib/out.pickle", "wb") as f:
                pickle.dump(self, f)
        except:
            print("There was an error storing the calibration data")
        # return status, cam, distortion, r, t
