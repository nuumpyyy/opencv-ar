import cv2 as cv

# Currently we are able to see matches between reference image and image in scene
def main():
    # initiate video capture
    cap = cv.VideoCapture(1)

    # load static reference image
    ref = cv.imread('surface/ref.png')
    # create SIFT feature detection object
    sift = cv.SIFT_create()
    # compute keypoints and descriptors of reference image
    kp_ref, des_ref = sift.detectAndCompute(ref, None)

    # create brute force matcher object with default params
    bf = cv.BFMatcher()

    while True:
        # read current frame
        ret, scene = cap.read()
        if not ret:
            print("Video capture unsuccessful.")
            break
        # compute keypoints and descriptors of card in scene
        kp_scene, des_scene = sift.detectAndCompute(scene, None)
        # find matches between reference image of card and card in scene
        matches = bf.match(des_ref, des_scene)
        # sort matches
        matches = sorted(matches, key=lambda x: x.distance)
        # draw and display matches
        scene = cv.drawMatches(ref, kp_ref, scene, kp_scene, matches[:10], 0, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("livestream!", scene)
        # press q to exit video capture
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # some procedural stuff
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()