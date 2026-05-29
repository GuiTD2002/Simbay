
import numpy as np
import argparse
import glob
import cv2


def find_chessboard_corners(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

def calibrate_camera(images, board_size, square_size):
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    for image in images:
        ret, corners = find_chessboard_corners(image, board_size)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            print("okay image - corners found")
        else:
            print("chessboard corners not found")

    gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def save_calibration_coefficients(path, ret, dist, mtx):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("mtx", mtx)
    cv_file.write("dist", dist)
    cv_file.write("ret", ret)
    cv_file.release()
    
def read_calibration_coefficients(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    ret = cv_file.getNode("ret").real()
    dist = cv_file.getNode("dist").mat()
    mtx = cv_file.getNode("mtx").mat()
    return ret, mtx, dist

def main():
    parser = argparse.ArgumentParser(description="Camera calibration")
    parser.add_argument("--image_dir", type=str, required=True, help="image directory path")
    parser.add_argument("--image_format", type=str, required=True,  help="image format, png/jpg")
    parser.add_argument("--prefix", type=str, required=True, help="image prefix")
    parser.add_argument("--square_size", type=float, required=False, help="chessboard square size")
    parser.add_argument("--width", type=int, required=False, help="chessboard width size, default is 9")
    parser.add_argument("--height", type=int, required=False, help="chessboard height size, default is 6")
    parser.add_argument("--save_file", type=str, required=True, help="YAML file to save calibration matrices")

    args = parser.parse_args()

    if args.square_size is None:
        square_size = 1.0  # Default square size
    else:
        square_size = args.square_size

    board_size = (args.width or 9, args.height or 6)

    images = glob.glob(f"{args.image_dir}/{args.prefix}*.{args.image_format}")
    images = [cv2.imread(image) for image in images]

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(images, board_size, square_size)
    save_calibration_coefficients(args.save_file, ret, dist, mtx)
    print("Calibration is finished. RMS:", ret)

if __name__ == "__main__":
    main()
    
    

