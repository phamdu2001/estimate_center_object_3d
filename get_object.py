import cv2
import numpy as np
import matplotlib.pyplot as plt
from get_mask import *

hh = 500
ww = 400

path = "D:/Study/Xulyanh/code/data/object.jpg"
image = cv2.imread(path)
        
# resize image
image = cv2.resize(image,(hh,ww))
img = np.copy(image)
#1. Tách màu sáng, giữ lại màu đen
# converting frame(image == BGR) to HSV(hue-saturation-value)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower_red = np.array([165, 65, 0])
# upper_red = np.array([255, 255, 255])

lower_red = np.array([36, 17, 0])
upper_red = np.array([88, 255, 255])
        
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

# # # draw points
# for point in approx[:3]:
#     point = point[0]; # drop extra layer of brackets
#     center = (int(point[0]), int(point[1]));
#     cv2.circle(img, center, 4, (150, 200, 0), -1);

center_bottom = (approx[2]+approx[4])/2
center_top = (approx[1]+approx[5])/2

path_board = "D:/Study/Xulyanh/code/data/back.jpg"
mask_board_chess, corners = detect_chess_board(path_board)
# for point in corners[:,0,:]:
#     cv2.circle(img, (int(point[0]),int(point[1])), 4, (255, 0, 0), -1)


near_point, near_point_index,arr_near,arr_point = point_near(center_bottom[0], corners[:,0,:])

for point in arr_point:
    cv2.circle(img, (int(point[0]),int(point[1])), 4, (255, 200, 0), -1);

center_object = (center_bottom[0] + center_top[0])/2

# cv2.circle(img, (int(center_bottom[0][0]),int(center_bottom[0][1])), 4, (150, 200, 0), -1);
# cv2.circle(img, (int(center_top[0][0]),int(center_top[0][1])), 4, (255, 200, 0), -1);
cv2.circle(img, (int(center_object[0]),int(center_object[1])), 4, (0, 0, 200), -1);

# x,y = hinhchieu(center_bottom[0], arr_point[1], arr_point[2], arr_point[3])
# # cv2.circle(img, (int(x),int(y)), 4, (0, 200, 255), -1)
# print(x,y)

h1 = hinhchieu(center_bottom[0], arr_point[1], arr_point[2], arr_point[3])
h2 = hinhchieu(center_bottom[0],arr_point[0], arr_point[3], arr_point[2])
# # print(h1)
# cv2.circle(img, (int(h1[0]),int(h1[1])), 4, (255, 200, 0), -1)
# cv2.circle(img, (int(h2[0]),int(h2[1])), 4, (255, 200, 0), -1)
x = arr_near[0][0]*2.75 + ti_le(arr_point[1],h1,arr_point[2])+2.75
y = arr_near[0][1]*2.75 + ti_le(h2,center_bottom[0],h1)+2.75

print(x,y)
# cv2.imshow("result", img)
# cv2.waitKey()

#doi thu tu truc cam
position_cam = np.array([11,22.5])

position_sample = np.array([5.584615384615384,5.628528777315784])
high_sample = np.array([44.598206241955516])
high_object = np.array([distance(center_bottom[0],center_top[0])])
position_object = np.array([x,y])
high_sample_real = 5

distance_cam_to_object = distance(position_cam,position_object)
distance_cam_to_sample = distance(position_cam,position_sample)

high_object_real = high_sample_real*high_object/(high_sample/distance_cam_to_sample*distance_cam_to_object)
print(high_object_real)

coordinates = f"({round(float(x),1)},{round(float(y),1)},{round(float(high_object_real/2),1)})"
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (int(center_object[0] + 10),int(center_object[1] - 10))
# fontScale
fontScale = 1
# Blue color in BGR
color = (0, 150, 0)
# Line thickness of 2 px
thickness = 2
# Using cv2.putText() method
image = cv2.putText(img, coordinates, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

cv2.imshow("result", image)
cv2.waitKey()