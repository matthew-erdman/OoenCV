import numpy as np
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.0.1/bin/tesseract'


def straightenCard(c, frame):
    # find rotated bounding box of card
    bounding = cv2.minAreaRect(c)
    h = int(bounding[1][0])
    w = int(bounding[1][1])
    box = np.float32(cv2.boxPoints(bounding))

    # straighten card
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    straightenedMatrix = cv2.getPerspectiveTransform(box, dst)
    straightened = cv2.warpPerspective(frame, straightenedMatrix, (w, h))
    w = straightened.shape[0]
    h = straightened.shape[1]

    # fix cards straightened with an angle less than 45 degrees
    if bounding[2] < 45.0:
        straightened = cv2.rotate(straightened, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return straightened


def readCards(frame, cnts):
    cards = []
    for (i, c) in enumerate(cnts):
        # straighten card
        straightened = straightenCard(c, frame)
        h = straightened.shape[0]
        w = straightened.shape[1]

        # focus on top left rank
        roi = straightened[0:int(h*0.3), 0:int(w*0.2)]

        # attempt single character ocr with whitelist
        char = pytesseract.image_to_string(roi, config="--psm 10 -c tessedit_char_whitelist=0123456789AJQK").strip()

        # no valid character found in top left, fallback to top right
        if not char:
            roi = straightened[0:int(h * 0.3), int(w * 0.8):w]
            char = pytesseract.image_to_string(roi, config="--psm 10 -c tessedit_char_whitelist=0123456789AJQK").strip()

        # parse ocr results
        if "A" in char:
            char = "Ace"
        elif "J" in char or "j" in char:
            char = "Jack"
        elif "Q" in char:
            char = "Queen"
        elif "K" in char or "k" in char:
            char = "King"
        elif char == "0":
            char = "10"
        cards.append(char)

        cv2.imshow("Straightened: " + str(i), straightened)
        cv2.imshow("Roi: " + str(i), roi)

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
    while True:
        ret, frame = video.read()
        # no frame received from camera
        if not ret:
            print("No frame received from camera - exiting...")
            break

        cv2.imshow('Video', frame)
        keypress = cv2.waitKey(1)

        # analyze selected frame
        if keypress == ord(" "):
            cv2.destroyAllWindows()
            # grayscale and blur frame
            processed = preprocess(frame)
            # find card border-like contours in the frame
            borders = findContours(processed)
            # detect card values
            cardVals = readCards(frame, borders)

            # display results
            cards = frame.copy()
            cv2.drawContours(cards, borders, -1, (0, 255, 0), 2)
            image = cv2.putText(cards, str(cardVals), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Cards', cards)
            if cv2.waitKey(0) in [ord("q"), ord("Q")]:
                break

        if keypress in [ord("q"), ord("Q")]:
            break

    video.release()
    cv2.destroyAllWindows()


main()