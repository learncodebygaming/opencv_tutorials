import cv2 as cv
import numpy as np
import os


# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Can use IMREAD flags to do different pre-processing of image files,
# like making them grayscale or reducing the size.
# https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('albion_cabbage.jpg', cv.IMREAD_UNCHANGED)

# There are 6 comparison methods to choose from:
# TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
# You can see the differences at a glance here:
# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
# Note that the values are inverted for TM_SQDIFF and TM_SQDIFF_NORMED
result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

# You can view the result of matchTemplate() like this:
#cv.imshow('Result', result)
#cv.waitKey()
# If you want to save this result to a file, you'll need to normalize the result array
# from 0..1 to 0..255, see:
# https://stackoverflow.com/questions/35719480/opencv-black-image-after-matchtemplate
#cv.imwrite('result_CCOEFF_NORMED.jpg', result * 255)

# Get the best match position from the match result.
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
# The max location will contain the upper left corner pixel position for the area
# that most closely matches our needle image. The max value gives an indication
# of how similar that find is to the original needle, where 1 is perfect and -1
# is exact opposite.
print('Best match top left position: %s' % str(max_loc))
print('Best match confidence: %s' % max_val)

# If the best match value is greater than 0.8, we'll trust that we found a match
threshold = 0.8
if max_val >= threshold:
    print('Found needle.')

    # Get the size of the needle image. With OpenCV images, you can get the dimensions 
    # via the shape property. It returns a tuple of the number of rows, columns, and 
    # channels (if the image is color):
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    # Calculate the bottom right corner of the rectangle to draw
    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    # Draw a rectangle on our screenshot to highlight where we found the needle.
    # The line color can be set as an RGB tuple
    cv.rectangle(haystack_img, top_left, bottom_right, 
                    color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)

    # You can view the processed screenshot like this:
    #cv.imshow('Result', haystack_img)
    #cv.waitKey()
    # Or you can save the results to a file.
    # imwrite() will smartly format our output image based on the extension we give it
    # https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
    cv.imwrite('result.jpg', haystack_img)

else:
    print('Needle not found.')
