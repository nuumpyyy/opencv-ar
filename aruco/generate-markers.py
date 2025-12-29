import cv2 as cv

# Create dictionary of markers
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
marker_size = 400

# Generate IDs
for marker_id in range(20):
    # generate marker
    img = cv.aruco.generateImageMarker(dictionary, marker_id, marker_size)
    print(img.shape)
    # save/write image
    cv.imwrite("marker_image{}.png".format(marker_id), img)

# Display marker images on windows
cv.imshow("Marker", img)
cv.waitKey(3000)