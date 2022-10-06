import numpy as np
import argparse
import cv2
import mahotas

def readCards(frame, cnts, padding):
    cards = []
    for (i, c) in enumerate(cnts):
        # find bounding box of each card
        (x, y, w, h) = cv2.boundingRect(c)
        # locate card with padding
        card = frame[y - padding:y + h + padding, x - padding:x + w + padding]
        cards.append(card)

    return cards


def findContours(frame):
    # find edges in frame
    edges = cv2.Canny(frame, 30, 150)
    # find outermost contours in frame, edges of each card
    (cnts, _) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    validBorders = []
    for c in cnts:
        # find bounding box of each card
        (x, y, w, h) = cv2.boundingRect(c)
        # ensure contour is large enough to be a card border
        if w > 200 and h > 300:
            validBorders.append(c)
    return validBorders


def preprocess(frame):
    # grayscale frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur frame
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
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

        # find card border-like contours in the frame
        borders = findContours(processed)
        cardVals = readCards(processed, borders, 20)
        for i, card in enumerate(cardVals):
            cv2.imshow("card " + str(i), card)
        cards = frame.copy()
        cv2.drawContours(cards, borders, -1, (0, 255, 0), 2)

        image = cv2.putText(cards, str(len(borders)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Cards', cards)
        if cv2.waitKey(1) != -1:
            break

    video.release()
    cv2.destroyAllWindows()


main()