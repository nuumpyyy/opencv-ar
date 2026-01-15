# opencv-ar
Real-time augmented reality application built with Python, NumPy, and OpenCV that renders .obj files onto ArUco markers

# Demo
[![Watch the video](https://img.youtube.com/vi/OksYdurXw8I/0.jpg)](www.youtube.com)

:33

# Usage
* Replace the images in `calibration/images` with camera calibration images taken from the camera which you wish to use to run the program
* Run `calibration/calibration.py` to obtain your camera's intrinsic matrix and distortion coefficients; modify `src/constants.py` accordingly
* On line 13 of `src/main.py` replace `'gaming-chair.obj'` with the name of the model you wish to render
* Run `python src/main.py` in a terminal session inside the project folder

# Next Steps
possible OpenGL (PyOpenGL) rendering??? (for more complex model textures)