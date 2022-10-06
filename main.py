import numpy as np
import argparse
import cv2
import mahotas

def findContours(frame):
    # find edges in frame
    edges = cv2.Canny(frame, 30, 150)
    # find outermost contours in frame, edges of each card
    (cnts, _) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cnts

def preprocess(frame):
    # grayscale frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur frame
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    return blurred


def main():
    # initialize video capture device
    video = cv2.VideoCapture(0)
    # video loop
    while (True):
        ret, frame = video.read()
        # no frame received from camera
        if not ret:
            print("No frame received from camera - exiting...")
            break

        # grayscale and blur frame
        processed = preprocess(frame)

        # find card-like contours in the frame
        cnts = findContours(processed)
        cards = frame.copy()
        cv2.drawContours(cards, cnts, -1, (0, 255, 0), 2)

        cv2.imshow('Cards', cards)
        if cv2.waitKey(1) != -1:
            break

    video.release()
    cv2.destroyAllWindows()


main()