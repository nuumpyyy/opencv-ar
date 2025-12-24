import numpy as np
import cv2 as cv
from calibration import in_mtx

# Currently we are able to see matches between reference image and image in scene
def main():
    # initiate video capture
    cap = cv.VideoCapture(1)

    # load static reference image
    ref = cv.imread('surface/ref.jpg')
    # convert to grayscale
    ref_gray = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
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

    print(homo_matrix, projection)

def projection_mtx(homo_matrix):
    in_mtx_inv = np.linalg.inv(in_mtx)
    extract = np.dot(homo_matrix, in_mtx_inv)
    return extract

if __name__ == "__main__":
    main()