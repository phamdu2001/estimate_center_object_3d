import cv2
import numpy as np
import matplotlib.pyplot as plt
from get_mask import *

def get_center_sample_object(path):
    hh = 500
    ww = 400

    # path = "D:/Study/Xulyanh/code/data/b2.jpg"
    image = cv2.imread(path)
            
    # resize image
    image = cv2.resize(image,(hh,ww))
    img = np.copy(image)
    #1. Tách màu sáng, giữ lại màu đen
    # converting frame(image == BGR) to HSV(hue-saturation-value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([165, 65, 0])
    # upper_red = np.array([255, 255, 255])

    lower_red = np.array([0, 79, 83])
    upper_red = np.array([255, 255, 255])
            
    # Morphological Transformations,Opening and Closing
    thresh = cv2.inRange(hsv,lower_red, upper_red)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(image, image, mask = mask)

    # contours # OpenCV 3.4, in OpenCV 2 or 4 it returns (contours, _)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    contour = contours[0]; # just take the first one

    # approx until 6 points
    num_points = 999999;
    step_size = 0.01;
    percent = step_size;
    while num_points >= 6:
        # get number of points
        epsilon = percent * cv2.arcLength(contour, True);
        approx = cv2.approxPolyDP(contour, epsilon, True);
        num_points = len(approx);

        # increment
        percent += step_size;

    # step back and get the points
    # there could be more than 6 points if our step size misses it
    percent -= step_size * 2;
    epsilon = percent * cv2.arcLength(contour, True);
    approx = cv2.approxPolyDP(contour, epsilon, True);

    # draw contour
    cv2.drawContours(img, [approx], -1, (0,0,200), 2);

    cv2.imshow("img",img)
    cv2.waitKey()
    # # draw points
    # for point in approx[:3]:
    #     point = point[0]; # drop extra layer of brackets
    #     center = (int(point[0]), int(point[1]));
    #     cv2.circle(img, center, 4, (150, 200, 0), -1);

    center_bottom = (approx[2]+approx[4])/2
    center_top = (approx[1]+approx[5])/2
    return center_bottom,center_top







ds = distance(o,s)
do = distance(o,ob)
high_object_real = high_sample_real * (high_sample/ds*do/high_object)