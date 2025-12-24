import numpy as np
import cv2 as cv
from math import *
from calibration import in_mtx

# Currently we are able to see matches between reference image and image in scene
def main():
    # initiate video capture
    cap = cv.VideoCapture(1)

    # load static reference image
    ref = cv.imread('surface/ref.jpg')
    # convert to grayscale
    ref_gray = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
    # load 3D model obj file
    # create SIFT feature detection object
    sift = cv.SIFT_create()
    # compute keypoints and descriptors of reference image
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)

    # create brute force matcher object with default params
    bf = cv.BFMatcher()

    while True:
        # read current frame
        ret, scene = cap.read()
        # convert scene image to grayscale
        scene_gray = cv.cvtColor(scene, cv.COLOR_BGR2GRAY)
        if not ret:
            print("Video capture unsuccessful.")
            break
        # compute keypoints and descriptors of card in scene
        kp_scene, des_scene = sift.detectAndCompute(scene_gray, None)
        # find matches between reference image of card and card in scene
        matches = bf.match(des_ref, des_scene)
        # sort matches
        matches = sorted(matches, key=lambda x: x.distance)

        # compute homography matrix if more than 10 matches are found
        if len(matches) > 10:
            # initialize homography matrix as None
            homo_matrix = None
            # extract locations of matched keypoints between the reference and scene images
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

            homo_matrix, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            # compute projection matrix and render 3D model if homography matrix is computed
            if homo_matrix is not None:
                projection = projection_mtx(homo_matrix)
                
        else:
            print(f"Not enough matches found — {len(matches)}/10.")

        # draw matches
        scene = cv.drawMatches(ref, kp_ref, scene, kp_scene, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv.imshow("livestream!", scene)
        # press q to exit video capture
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # some procedural stuff
    cap.release()
    cv.destroyAllWindows()

    print(projection)

# Calculate projection matrix
def projection_mtx(homo_matrix):
    # compute inverse of intrinsic camera matrix
    in_mtx_inv = np.linalg.inv(in_mtx)
    # extract [R1 R2 t]
    extract = np.dot(homo_matrix * (-1), in_mtx_inv)
    # extract individual rotation and translation vectors
    R1 = extract[:, 0]
    R2 = extract[:, 1]
    t = extract[:, 2]

    # normalize rotation vectors
    R1_norm = np.linalg.norm(R1, 2)
    R2_norm = np.linalg.norm(R2, 2)

    # update values of R1 and R2 accordingly
    l = sqrt(R1_norm*R2_norm)
    R1 = np.divide(R1, l)
    R2 = np.divide(R2, l)

    c = np.add(R1, R2)
    p = np.cross(R1, R2)
    d = np.cross(c, p)

    # perform transformation operations using the values of c and d
    R1 = np.add((np.divide(c, np.linalg.norm(c, 2))), (np.divide(d, np.linalg.norm(d, 2))))
    R1 = np.dot(R1, 1 / sqrt(2))
    R2 = np.subtract((np.divide(c, np.linalg.norm(c, 2))), (np.divide(d, np.linalg.norm(d, 2))))
    R2 = np.dot(R2, 1 / sqrt(2))
    R3 = np.cross(R1, R2)

    # combine vectors to form extrinsic camera matrix
    projection = np.stack((R1, R2, R3, t), axis=1)
    # multiply intrinsic and extrinsic matrices to obtain full camera projection matrix
    projection = np.dot(in_mtx, projection)

    return projection

def render(frame, obj, projection, ref_shape):
    vertices = np.array(list(obj.vertices), dtype=np.float32)
    h, w = ref_shape[:2]

    scale = 3
    scale_matrix = np.eye(3) * scale

    for face in obj.faces:
        face_vertices, _, _, material = face

        pts = np.array([vertices[v-1] for v in face_vertices])
        pts = pts @ scale_matrix

        # center model on reference image
        pts[:, 0] += w / 2
        pts[:, 1] += h / 2

        pts = pts.reshape(-1, 1, 3)
        dst = cv.perspectiveTransform(pts, projection)
        imgpts = np.int32(dst)

        # get material color
        mtl = obj.mtl.get(material, {})
        if 'Kd' in mtl:
            color = tuple(int(c * 255) for c in mtl['Kd'])[::-1]
        else:
            color = (150, 150, 150)

        cv.fillConvexPoly(frame, imgpts, color)

    return frame

if __name__ == "__main__":
    main()