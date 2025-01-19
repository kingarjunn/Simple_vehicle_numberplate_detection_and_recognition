import os
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


from skimage.color.rgb_colors import green

cap = cv2.VideoCapture(0) # 0 for camera
while True:
    success, img = cap.read()
    cv2.imshow('img1',img)
    cv2.waitKey(1)
    #img = captureScreen()
    #un comment the below line if camera takes high pixel images/video
    #imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    if len(location)>0:
        try:
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)
            plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
            #plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            #cv2.imshow('img',cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
            #plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_image)
            img0=img
            #print(result)
            macer =20
            spacer = 100
            for detection in result:
                top_left = tuple(detection[0][0])
                bottom_right = tuple(detection[0][2])
                text = detection[1]
                img1 = cv2.rectangle(img0, x1+top_left, y1+bottom_right, (0, 255, 0), 3)
                img1 = cv2.putText(img1, text, (macer, spacer), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                img0=img1
                spacer += 15
                macer += 20
            cv2.imshow('img', img0)
            h, w, c = cropped_image.shape
            cv2.waitKey(1)
        except Exception as ex:
            print(str(ex))
