import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
wincap = WindowCapture('Albion Online Client')
# initialize the Vision class
vision_limestone = Vision('albion_limestone_edges.jpg')
# initialize the trackbar window
vision_limestone.init_control_gui()

# limestone HSV filter
hsv_filter = HsvFilter(0, 180, 129, 15, 229, 243, 143, 0, 67, 0)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    # pre-process the image
    processed_image = vision_limestone.apply_hsv_filter(screenshot)

    # do edge detection
    edges_image = vision_limestone.apply_edge_filter(processed_image)

    # do object detection
    #rectangles = vision_limestone.find(processed_image, 0.46)

    # draw the detection results onto the original image
    #output_image = vision_limestone.draw_rectangles(screenshot, rectangles)

    # keypoint searching
    keypoint_image = edges_image
    # crop the image to remove the ui elements
    x, w, y, h = [200, 1130, 70, 750]
    keypoint_image = keypoint_image[y:y+h, x:x+w]

    kp1, kp2, matches, match_points = vision_limestone.match_keypoints(keypoint_image)
    match_image = cv.drawMatches(
        vision_limestone.needle_img, 
        kp1, 
        keypoint_image, 
        kp2, 
        matches, 
        None)

    if match_points:
        # find the center point of all the matched features
        center_point = vision_limestone.centeroid(match_points)
        # account for the width of the needle image that appears on the left
        center_point[0] += vision_limestone.needle_w
        # drawn the found center point on the output image
        match_image = vision_limestone.draw_crosshairs(match_image, [center_point])

    # display the processed image
    cv.imshow('Keypoint Search', match_image)
    cv.imshow('Processed', processed_image)
    cv.imshow('Edges', edges_image)
    #cv.imshow('Matches', output_image)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
