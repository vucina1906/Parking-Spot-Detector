Purpose

The purpose of this project was to create a parking spot detector that can be connected to a live camera feed and track whether there are free parking spaces and how many. To differentiate whether a parking space is occupied or not, I used image transformation techniques, which I found to be much more efficient than advanced object detection algorithms like the latest versions of YOLO. Given the nature of parking cameras, which are usually static, this technique—using image transformations and non-zero pixel counting—provides accurate results with fast response times.

Project Overview

The first part of the project involved finding and saving the exact coordinates for each parking spot. The code for this process is stored in find_spot_coordinates.py. After running this script, you will have the opportunity to choose the coordinates interactively. Each time you select four coordinates for a parking spot, the program will draw a rectangle around the spot so you can immediately verify if you are satisfied with the result. After finishing the selection, press "ESC" to save all the coordinates to a parking_spots.txt file within your project folder.

Afterward, you can run the main.py script. This script dynamically loads the coordinates from the parking_spots.txt file and utilizes them in the code. The example presented in this project was demanding, as the video I used was recorded from a drone, which is not as static as the parking cameras typically used. In certain parts of the video, the drone's slight movements caused issues with pixel counting. To address this, I created separate conditions for specific parking spots that were most affected by the camera's movement.

Usage

You can adapt this code for your own projects. If you have more stable recordings from static cameras, you’ll only need to find the coordinates and select a unique threshold for all the parking spots. If you have situations where trees or other objects might interfere with the detection (e.g., due to weather conditions), you can also create a similar solution to mine by defining separate conditions for specific parking spots.