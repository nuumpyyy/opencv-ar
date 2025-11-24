import numpy as np
import cv2 as cv
import os
import math
from calibration import in_mtx
from objloader import *

MIN_MATCHES = 10
COLOR = (252, 225, 231)

# Estimate calibration matrix if calibration images fail to return result
if in_mtx is None:
    in_mtx = np.array([
        [1536, 0, 960],
        [0, 1536, 540],
        [0, 0, 1]
    ])

def main():
    # initiate video capture
    cap = cv.VideoCapture(1)
    # load reference card surface image
    dir_name = os.getcwd()
    ref = cv.imread(os.path.join(dir_name, 'surface/ref.png'), 0) # static reference image
    # load 3D Pokémon model from OBJ file
    model = OBJ(os.path.join(dir_name, 'projection/fox.obj'), swapyz=True)

    while True:
        # extract current video frame
        ret, scene = cap.read()
        if not ret:
            print("Video capture unsuccessful.")
            return
        
        cv.imshow("livestream", scene)

        # obtain keypoints from and matches between reference image and video capture scene
        matches, kp_ref, kp_scene = detect_and_match(ref, scene)
        
        # estimate homography if enough matches are found
        if len(matches) >= MIN_MATCHES:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            homo = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)[0]

            # draw rectangle around reference surface found in scene
            h, w = ref.shape
            pts = np.float32([0, 0], [0, h-1], [w-1, h-1], [w-1, 0]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, homo)
            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

            # if homography matrix was found render 3D Pokémon model onto video capture scene
            if homo is not None:
                proj = projection_matrix(in_mtx, homo)
                scene = render(scene, ref, model, proj, COLOR)

        else:
            print("Not enough matches found.")

        # command to exit program
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

def detect_and_match(ref, scene):
    # initiate SIFT keypoint detector
    sift = cv.SIFT_create()
    # compute card surface keypoints and descriptors with SIFT
    kp_ref, des_ref = sift.detectAndCompute(ref, None)
    # compute scene keypoints and descriptors with SIFT
    kp_scene, des_scene = sift.detectAndCompute(scene, None)
    # create BF matcher object with default params
    bf = cv.BFMatcher()
     # match reference descriptors with scene descriptors
    matches = bf.match(des_ref, des_scene)
    # sort matches in order of distance
    matches = sorted(matches, key=lambda x: x.distance)
    # only keep good matches according to Lowe's ratio test
    #for m, n in matches:
        #if m.distance > 0.7*n.distance:
            #matches.remove(m)
    return matches, kp_ref, kp_scene

def projection_matrix(in_mtx, homo):
    # compute rotation and translation vectors
    r_and_t = np.dot(np.linalg.inv(in_mtx, homo*(-1)))
    col_1 = r_and_t[:, 0]
    col_2 = r_and_t[:, 1]
    col_3 = r_and_t[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2)*np.linalg.norm(col_2, 2))
    R1 = col_1 / l
    R2 = col_2 / l
    t = col_3 / l
    # compute more accurate estimations of rotation and translation vectors
    c = R1 + R2
    p = R1 * R2
    d = c * p
    # calculate value of R3
    new_R1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    new_R2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    R3 = np.cross(new_R1, new_R2)
    # compute extrinsic matrix
    ex_mtx = np.stack((R1, R2, R3, t)).T
    proj = np.dot(in_mtx, ex_mtx)
    return proj

def render(scene, ref, model, proj, color):
    # obtain dimensions of card surface and 3D vertices of model
    h, w = ref.shape
    vertices = model.vertices
    mtx = np.eye(3)*3
    # project camera calibration matrix onto each 3D point of model
    for face in model.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, mtx)
        # render model in middle of card
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv.perspectiveTransform(points.reshape(-1, 1, 3), proj)
        scene_pts = np.int32(dst)
        cv.fillConvexPoly(scene, scene_pts, color)
    return scene

if __name__ == "__main__":
    main()