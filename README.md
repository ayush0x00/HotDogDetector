# CPPND: Capstone

This is a repo for the Capstone project in the [Udacity C++ Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213).

In this project, I finished the Hot Dog Detector using OpenCV and yolov3, you can simply detect all the hot dogs in a picture.

This application include 3 progresses:
1. Initialize a YoloNet instance to prepare for detection.
2. Read the input image from default path or argument.
3. Show the hot dog detection results.

The necessary rubric my submission satisfies:
1. The project demonstrates an understanding of C++ functions and control structures.
2. The project reads data from a file and process the data, or the program writes data to a file.
3. The project accepts user input and processes the input.
4. The project uses Object Oriented Programming techniques.
5. Classes use appropriate access specifiers for class members.
6. The project makes use of references in function declarations.

## Dependencies for Running Locally
* cmake >= 3.7
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac)
  * Linux: make is installed by default on most Linux distros
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
* OpenCV >= 4.1
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)

## Basic Build Instructions

1. Clone this repo.
2. Compile: `cmake . && make`
3. Run it: `./HotDogDetector <your_image.jpg>`.
