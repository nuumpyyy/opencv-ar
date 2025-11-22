import numpy as np
import cv2 as cv
import os
import math

MIN_MATCHES = 10

def main():
    dir_name = os.getcwd()
    ref = cv.imread(os.path.join(dir_name, 'surface/ref.png'), 0) # static reference image
    scene = cv.imread(os.path.join(dir_name, 'surface/scene.png'), 0) # one possible static video frame of surface
    matches, result = detect_and_match(ref, scene)
    # compute homography if enough matches are found
    if len(matches) >= MIN_MATCHES:
        cv.imshow('result', result)
        cv.waitKey(0)
    else:
        print("Not enough matches ({m}/{M}) found.".format(m=matches, M=MIN_MATCHES))

# Returns matches between features in reference image and each frame from video capture
def detect_and_match(ref, scene):
    # initiate SIFT keypoint detector
    sift = cv.SIFT_create()
    # compute keypoints and descriptors with SIFT
    kp_ref, des_ref = sift.detectAndCompute(ref, None)
    kp_scene, des_scene = sift.detectAndCompute(scene, None)
    # create BF matcher object with default params
    bf = cv.BFMatcher()
    # match reference descriptors with scene descriptors
    matches = bf.match(des_ref, des_scene)
    # sort matches in order of distance
    matches = sorted(matches, key=lambda x: x.distance)
    # draw first 10 matches
    result = cv.drawMatches(ref, kp_ref, scene, kp_scene, matches[:MIN_MATCHES], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matches, result

if __name__ == "__main__":
    main()