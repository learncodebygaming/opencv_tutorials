import cv2 as cv
import numpy as np
import os
from time import time
from modules.windowcapture import WindowCapture
from modules.vision import Render, Search

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
wincap = WindowCapture("Albion Online Client")
# initialize the Vision class
vision_limestone = Search(
    "assets/albion/limestone",
    threshold=0.98,
    method=cv.TM_CCORR_NORMED,
    debug="rectangles",
)
# init renderer
renderer = Render(debug="rectangles")

"""
# https://www.crazygames.com/game/guns-and-bottle
wincap = WindowCapture()
vision_gunsnbottle = Vision('gunsnbottle.jpg')
"""

wincap.start()
screenshot, width, height = wincap.get_screen()
vision_limestone.setHaystack(screenshot)
renderer.setHaystack(screenshot)
vision_limestone.start()
renderer.start()

while True:
    rectangles = []
    if vision_limestone.stopped or renderer.stopped:
        renderer.stop()
        vision_limestone.stop()
        wincap.stop()
        break

    # get an updated image of the game
    screenshot, width, height = wincap.get_screen()
    vision_limestone.setHaystack(screenshot)
    renderer.setHaystack(screenshot)

    # display the processed image
    points = vision_limestone.getPoints()
    rectangles = vision_limestone.getRectangles()
    renderer.setPoints(points)
    renderer.setRectangles(rectangles)

    # vision_limestone.increment()
    # renderer.increment()

    # Some stats
    print(
        "Search {} iterations/sec, Render {} iterations/sec".format(
            vision_limestone.countsPerSec(), renderer.countsPerSec()
        )
    )

print("Done.")
