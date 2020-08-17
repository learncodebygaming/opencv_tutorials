import os, time
import cv2 as cv
import numpy as np
from threading import Thread, Lock
from datetime import datetime


class Needle:
    # properties
    needle_img = None
    needle_w = 0
    needle_h = 0

    # constructor
    def __init__(self, needle_img_path):
        # load the image we're trying to match
        # https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
        self.needle_img = cv.imread(needle_img_path, cv.IMREAD_GRAYSCALE)
        print(needle_img_path)

        # Save the dimensions of the needle image
        self.needle_w = self.needle_img.shape[1]
        self.needle_h = self.needle_img.shape[0]


class Search:
    # properties
    stopped = True
    current = 0
    haystack_img = None
    method = None
    needles = []
    points = []
    rectangles = []
    results = []

    # constructor
    def __init__(
        self, needle_img_path, threshold=0.5, method=cv.TM_CCOEFF_NORMED, debug=None
    ):
        # load the image we're trying to match and init
        # https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
        self.debug = debug
        self.mut = Lock()
        for image in os.listdir(needle_img_path):
            self.needles.append(Needle(f"{needle_img_path}/{image}"))

        # There are 6 methods to choose from:
        # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
        self.method = method
        self.threshold = threshold

        # iteration counter
        self._start_time = None
        self._num_occurrences = 0

    # Start the async thread that is matching images
    def start(self):
        self.stopped = False
        thrd = Thread(target=self.find)
        thrd.start()
        # init iterator
        self._start_time = datetime.now()
        return True

    # Stop the async thread that is matching images
    def stop(self):
        self.stopped = True

    # Iterator occurance counter
    def increment(self):
        self._num_occurrences += 1

    # Iteratoions per sec
    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0

    # Set haystack image from outside of the thread
    def setHaystack(self, haystack_img):
        self.mut.acquire()
        self.haystack_img = haystack_img
        self.mut.release()

    # Match template with haystack
    def find(self):
        while not self.stopped:
            self.mut.acquire()
            img_gray = cv.cvtColor(self.haystack_img, cv.COLOR_BGR2GRAY)
            self.mut.release()
            rectangles = []

            for needle in self.needles:
                # run the OpenCV algorithm
                result = cv.matchTemplate(img_gray, needle.needle_img, self.method)

                # Get the all the positions from the match result that exceed our threshold
                locations = np.where(result >= self.threshold)
                # locations = list(zip(*locations[::-1]))
                # print(locations)

                # You'll notice a lot of overlapping rectangles get drawn. We can eliminate those redundant
                # locations by using groupRectangles().
                # First we need to create the list of [x, y, w, h] rectangles
                for loc in zip(*locations[::-1]):
                    rect = [int(loc[0]), int(loc[1]), needle.needle_w, needle.needle_h]
                    # Add every box to the list twice in order to retain single (non-overlapping) boxes
                    rectangles.append(rect)
                    rectangles.append(rect)

            # Apply group rectangles.
            # The groupThreshold parameter should usually be 1. If you put it at 0 then no grouping is
            # done. If you put it at 2 then an object needs at least 3 overlapping rectangles to appear
            # in the result. I've set eps to 0.5, which is:
            # "Relative difference between sides of the rectangles to merge them into a group."
            rectangles, weights = cv.groupRectangles(
                rectangles, groupThreshold=1, eps=0.5
            )
            # print(rectangles)

            self.mut.acquire()
            self.rectangles = rectangles
            self.mut.release()

            # Increments occurance counter
            self.increment()

    # Get click points from outside of thread
    def getPoints(self):
        self.mut.acquire()
        rectangles = self.rectangles
        self.mut.release()
        points = []
        if not isinstance(rectangles, int):
            # Loop over all the rectangles
            for (x, y, w, h) in rectangles:

                # Determine the center position
                center_x = x + int(w / 2)
                center_y = y + int(h / 2)
                # Save the points
                points.append((center_x, center_y))

        return points

    # Get tectangles from outside of thread
    def getRectangles(self):
        self.mut.acquire()
        rectangles = self.rectangles
        self.mut.release()
        return rectangles if not isinstance(rectangles, tuple) else []


class Render:
    # properties
    stopped = True
    haystack_img = None
    rectangles = []
    points = []

    # constructor
    def __init__(self, debug=None):
        # init lock and debug mode
        self.debug = debug
        self.mut = Lock()
        # interation counter
        self._start_time = None
        self._num_occurrences = 0

    # Start the async thread that is rendering images
    def start(self):
        self.stopped = False
        thrd = Thread(target=self.draw)
        thrd.start()
        # init iterator
        self._start_time = datetime.now()
        return True

    # Stop the async thread that is rendering images
    def stop(self):
        self.stopped = True

    # Iterator occurance counter
    def increment(self):
        self._num_occurrences += 1

    # Iteratoions per sec
    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time if elapsed_time > 0 else 0

    def setHaystack(self, haystack_img):
        self.mut.acquire()
        self.haystack_img = haystack_img
        self.mut.release()

    def setPoints(self, points):
        self.mut.acquire()
        self.points = points
        self.mut.release()

    def setRectangles(self, rectangles):
        self.mut.acquire()
        self.rectangles = rectangles
        self.mut.release()

    def draw(self):
        while not self.stopped:
            while self.haystack_img is None:
                pass
            self.mut.acquire()
            haystack_img = self.haystack_img
            points = self.points
            rectangles = self.rectangles
            self.mut.release()
            if isinstance(rectangles, list) or isinstance(points, list):
                # print('Found needle.')
                line_color = (0, 255, 0)
                line_type = cv.LINE_4
                marker_color = (255, 0, 255)
                marker_type = cv.MARKER_CROSS

                # Loop over all the rectangles
                if self.debug == "rectangles":
                    for (x, y, w, h) in rectangles:
                        # Determine the box position
                        top_left = (x, y)
                        bottom_right = (x + w, y + h)
                        # Draw the box
                        cv.rectangle(
                            haystack_img,
                            top_left,
                            bottom_right,
                            color=line_color,
                            lineType=line_type,
                            thickness=2,
                        )
                elif self.debug == "points":
                    for (center_x, center_y) in points:
                        # Draw the center point
                        cv.drawMarker(
                            haystack_img,
                            (center_x, center_y),
                            color=marker_color,
                            markerType=marker_type,
                            markerSize=40,
                            thickness=2,
                        )

            if self.debug:
                cv.imshow("Matches", haystack_img)

            if cv.waitKey(5) == ord("q"):
                self.stopped = True

            # Increments occurance counter
            self.increment()
