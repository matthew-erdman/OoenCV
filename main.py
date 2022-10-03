import numpy as np
import argparse
import cv2

def main():
    # Build argument parser and load image from user
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])

    cv2.imshow("Image", image)
    cv2.waitKey(0)

main()