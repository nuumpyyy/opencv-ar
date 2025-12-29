import numpy as np
import cv2 as cv
from constants import *

def main():
    # initiate video capture
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 20)
    if not cap.isOpened():
        print("Video capture unsuccessful.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break
        # detect whether or not ArUco markers have been found
        found, corners = detect_aruco_markers(frame)
        # render 3D model onto frame if ArUco marker found
        if found:
            # compute rotation and translation vectors
            rvecs, tvecs = get_rvecs_and_tvecs(20.2, corners) # side length of marker is about 20.2 cm
            # calculate projection matrix consisting of intrinsic and extrinsic camera parameters
            projection = get_projection_matrix(rvecs, tvecs)
            # render 3D model
            frame = render(frame, projection, corners)
        # display frame
        cv.imshow("Stream", frame)
        # press 'q' to exit loop and end video capture
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

# Returns Boolean indicating whether or not ArUco marker has been found, as well as corners and marker ids if found
def detect_aruco_markers(frame):
    found = False
    # initialize dictionary of ArUco markers
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    # convert image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, marker_ids, _ = cv.aruco.detectMarkers(gray, aruco_dict)
    if marker_ids is not None and corners is not None:
        found = True
    return found, corners

# If ArUco markers are found, use them to find the rotation and translation vectors
# Rvecs and tvecs calculation from https://stackoverflow.com/questions/75750177/solve-pnp-or-estimate-pose-single-markers-which-is-better/75871586#75871586
def get_rvecs_and_tvecs(marker_size, corners):
    # use size (side length in cm) of square marker to estimate real world points
    marker_pts = np.array([[-marker_size / 2, marker_size / 2, 0],
                            [marker_size / 2, marker_size / 2, 0],
                            [marker_size / 2, -marker_size / 2, 0],
                            [-marker_size / 2, - marker_size / 2, 0]], dtype=np.float32)
    rvecs = []
    tvecs = []
    for corner in corners:
        _, R, t = cv.solvePnP(marker_pts, corner, IN_MTX_OPTIMAL, DIST_COEFFS, False, cv.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
    return rvecs, tvecs

def get_projection_matrix(rvecs, tvecs):
    # use cv.Rodrigues() method to obtain 3x3 rotation matrix from 1x3 rotation vector
    rmtx, _ = cv.Rodrigues(rvecs[0])
    # construct extrinsic camera parameters matrix from rotation matrix and translation vectors
    ex_mtx = np.array([[rmtx[0,0], rmtx[0,1], rmtx[0,2], tvecs[0][0,0]],
                       [rmtx[1,0], rmtx[1,1], rmtx[1,2], tvecs[0][1,0]],
                       [rmtx[2,0], rmtx[2,1], rmtx[2,2], tvecs[0][2,0]]])
    #ex_mtx = np.array([[rmtx[0,0], rmtx[0,1], rmtx[0,2], tvecs[0][0,0]],
                       #[rmtx[1,0], rmtx[1,1], rmtx[1,2], tvecs[0][1,0]],
                       #[rmtx[2,0], rmtx[2,1], rmtx[2,2], tvecs[0][2,0]],
                       #[0.0, 0.0, 0.0, 1.0]])
    return np.dot(IN_MTX_OPTIMAL, ex_mtx)

# Returns frame with ArUco marker blacked out
def render(frame, projection, corners):
    for corner in corners:
        cv.polylines(frame, [corner.astype(np.int32)], True, (0, 255, 255), 3, cv.LINE_AA)
    return frame

if __name__ == "__main__":
    main()

#sample_frame = cv.imread('aruco/sample-input1.png')
#found, corners = detect_aruco_markers(sample_frame)
#if found:
    #rvecs, tvecs = get_rvecs_and_tvecs(20.2, corners)
    #print(rvecs, tvecs)
    #print(np.shape(rvecs), np.shape(tvecs))

    #projection = get_projection_matrix(rvecs, tvecs)
    #print(projection)
    #print(np.shape(projection))

    #sample_frame = render(sample_frame, projection, corners)
#cv.imshow("Sample frame", sample_frame)
#cv.waitKey(0)
#cv.destroyAllWindows()