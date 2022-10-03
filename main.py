import numpy as np
import argparse
import cv2

def main():
    # initialize video capture device
    video = cv2.VideoCapture(0)

    # video loop
    while (True):

        ret, frame = video.read()
        if not ret:
            print("No frame received")
            break

        cv2.imshow('Cards', frame)
        if cv2.waitKey(1) != -1:
            break

    video.release()
    cv2.destroyAllWindows()


main()