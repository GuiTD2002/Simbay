import os
import cv2
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
from camera_calib.calibrate import read_calibration_coefficients
import math
import scipy.io
import time
import platform

class visFTDriver:
    def __init__(self) -> None:
        self.arducam = None
        self.mtx = None
        self.dist = None
        self.offset = np.zeros(6)
        self.history = []

        try:
            model = scipy.io.loadmat('camera_calib/calmat.mat')
            self.W1 = model['W1_stack']
            self.b1 = model['b1_stack']
            self.W2 = model['W2_stack']
            self.b2 = model['b2_stack']
            self.mu = model['mu'].flatten()
            self.sigma = model['sigma'].flatten()
        except FileNotFoundError:
            print("Error: Calibration matrix not found.")

    def start(self):
        calibFile="camera_calib/arducam.yaml"
        found_sensor=False
        is_windows = (platform.system() == 'Windows')
        backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_V4L2
        exposure_val = -12 if is_windows else 1
        for index in range(5):
            temp_cap = cv2.VideoCapture(index, backend)
            if not temp_cap.isOpened():
                temp_cap.release()
                continue
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1279)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
            actual_w = temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if int(actual_w) == 1280 and int(actual_h) == 800:
                self.arducam = temp_cap
                found_sensor = True
                break
            else:
                temp_cap.release()
        if not found_sensor:
            raise RuntimeError("Error: VIS_FT sensor not found...")

        if not is_windows:
            self.arducam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.arducam.set(cv2.CAP_PROP_BRIGHTNESS, -20)        

        self.arducam.set(cv2.CAP_PROP_FPS, 150)
        self.arducam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.arducam.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

        ret,self.mtx,self.dist=read_calibration_coefficients(calibFile)

        time.sleep(2)

        for _ in range(5):
            self.arducam.read()

        return True
    
    def zero(self, N=500):
        offset_values = np.zeros(6)
        nn = 0
        while nn < N:
            readsensor, _ = self.read(raw_data=False, window_size=1, drift_correct=False)
            if not any(np.isnan(readsensor)):
                offset_values += readsensor
                nn += 1
        self.offset = offset_values / N
        self.history = [np.zeros(6) for _ in range(N)]
        return True

    def shutdown(self):
        self.arducam.release()
        print('VISFTDriver: Exiting')

    def read(self, raw_data=False, window_size=30, drift_correct=True):
        ret_val, frame = self.arducam.read()

        if not ret_val or frame is None:
            return self.read(raw_data=raw_data, window_size=window_size)
            
        frame=cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ROIS = [
                (530, 20, 300, 300),
                (40, 380, 300, 300),
                (770, 490, 300, 300)
                ]

        roi_pixels = []
        for (x, y, w, h) in ROIS:
            roi = gray[y:y+h, x:x+w]
            roi = self.downsample_roi(roi)
            roi_pixels.append(roi.flatten())

        visft_data = np.zeros(6)

        if raw_data == True:
            visft_data = np.concatenate((roi_pixels), axis=0)
        
        else:
            pix_data = np.concatenate(roi_pixels, axis=0)
            x_norm = (pix_data - self.mu) / self.sigma

            hidden_layer = np.matmul(x_norm, self.W1) + self.b1
            hidden_layer = np.maximum(0, hidden_layer)

            visft_data = np.matmul(hidden_layer, self.W2) + self.b2
            visft_data = visft_data.flatten()

            visft_data[2] = -visft_data[2]
            visft_data[5] = -visft_data[5]
            visft_data[3] -= 0.15 * visft_data[1]
            visft_data[4] += 0.15 * visft_data[0]
        
            if drift_correct:
                alpha = 0.01
                tared_check = visft_data - self.offset
                if (np.linalg.norm(tared_check[0:3]) < 5.0) and (np.linalg.norm(tared_check[3:6]) < 0.1):
                    self.offset = (1 - alpha) * self.offset + alpha * visft_data
            
            visft_data -= self.offset

            if len(self.history) > window_size:
                self.history.pop(0)

            self.history.append(visft_data.copy())
            window = min(window_size, len(self.history))
            visft_data = np.mean(self.history[-window:], axis=0)

        return visft_data.tolist(),frame

    def downsample_roi(self, roi, new_size=(6,6)):
        h, w = roi.shape
        new_h, new_w = new_size

        factor_h = h // new_h
        factor_w = w // new_w

        roi_small = roi.reshape(new_h, factor_h, new_w, factor_w).mean(axis=(1,3))
        return roi_small
