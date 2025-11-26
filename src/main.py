# Testing testing rahhhhh
import cv2 as cv

# currently we are able to see matches between reference image and image in scene
def main():
    cap = cv.VideoCapture(1)

    ref = cv.imread('surface/ref.png')
    sift = cv.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(ref, None)
    bf = cv.BFMatcher()

    while True:
        ret, scene = cap.read()
        if not ret:
            print("Video capture unsuccessful.")
            break
        kp_scene, des_scene = sift.detectAndCompute(scene, None)
        matches = bf.match(des_ref, des_scene)
        matches = sorted(matches, key=lambda x: x.distance)
        scene = cv.drawMatches(ref, kp_ref, scene, kp_scene, matches[:10], 0, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("livestream!", scene)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()