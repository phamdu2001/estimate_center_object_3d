import cv2
import numpy as np
import matplotlib.pyplot as plt
from get_mask import *

# path = "D:/Study/Xulyanh/code/data/b1_.jpg"
# hh = 500
# ww = 400
# image = cv2.imread(path)
        
# # resize image
# image = cv2.resize(image,(hh,ww))

# #1. Tách màu sáng, giữ lại màu đen
# # converting frame(image == BGR) to HSV(hue-saturation-value)
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower_red = np.array([0, 0, 62])
# upper_red = np.array([255, 255, 255])
        
# # Morphological Transformations,Opening and Closing
# thresh = cv2.inRange(hsv,lower_red, upper_red)
# kernel = np.ones((5,5),np.uint8)
# mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# result = cv2.bitwise_and(image, image, mask = mask)

# #2. lấy vùng bàn cờ
# contours = cv2.findContours(255 - mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # print(contours)
# contours = contours[0] if len(contours) == 2 else contours[1]
# mask_0 = np.zeros((ww,hh), dtype=np.uint8)
# for cntr in contours:
#     cv2.drawContours(mask_0, [cntr], 0, (255,255,255), -1)
# result = cv2.bitwise_and(image, image, mask=mask_0)

# #3. Tách trắng lấy đen loại bỏ viền bàn cờ
# hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
# lower_red = np.array([0, 0, 0])
# upper_red = np.array([255, 255, 73])
        
# # Morphological Transformations,Opening and Closing
# thresh = cv2.inRange(hsv,lower_red, upper_red)
# kernel = np.ones((10,10),np.uint8)

# #4. khử nhiễu
# mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# image = np.zeros((ww,hh), dtype=np.uint8)
# image += 255
# result = cv2.bitwise_and(image, image, mask = mask)

# corners = 0
# mask_0 = mask
# #5. Lấy góc ô cờ
# found, corners = cv2.findChessboardCorners(result, (7, 7),flags=cv2.CALIB_CB_FAST_CHECK)
# # print(np.shape(corners))

# # corner_lt = (corners[0]-corners[16])

# img = cv2.imread(path)
# img = cv2.resize(img,(hh,ww))
# img2 = np.copy(img)  # Make a copy of original img as img2

# for corner in corners:
#     # print(corner[0])
#     coord = (int(corner[0][0]), int(corner[0][1]))
#     # print(coord)
#     cv2.circle(img=img2, center=coord, radius=2, color=(255, 0, 0), thickness=2)


# cv2.imshow("result", img2)
# cv2.waitKey()


hh = 500
ww = 400

path = "D:/Study/Xulyanh/code/data/b1.jpg"
image = cv2.imread(path)
        
# resize image
image = cv2.resize(image,(hh,ww))
img = np.copy(image)
#1. Tách màu sáng, giữ lại màu đen
# converting frame(image == BGR) to HSV(hue-saturation-value)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower_red = np.array([165, 65, 0])
# upper_red = np.array([255, 255, 255])

lower_red = np.array([30, 9, 78])
upper_red = np.array([87, 255, 255])
        
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

center_bottom = (approx[2]+approx[4])/2
center_top = (approx[1]+approx[5])/2

path_board = "D:/Study/Xulyanh/code/data/b1_.jpg"
mask_board_chess, corners = detect_chess_board(path_board)
# for point in corners[:,0,:]:
#     cv2.circle(img, (int(point[0]),int(point[1])), 4, (255, 0, 0), -1)


near_point, near_point_index,arr_near,arr_point = point_near(center_bottom[0], corners[:,0,:])

for point in arr_point:
    cv2.circle(img, (int(point[0]),int(point[1])), 4, (255, 200, 0), -1);

cv2.circle(img, (int(center_bottom[0][0]),int(center_bottom[0][1])), 4, (150, 200, 0), -1);
cv2.circle(img, (int(center_top[0][0]),int(center_top[0][1])), 4, (255, 200, 0), -1);

h1 = hinhchieu(center_bottom[0], arr_point[1], arr_point[2], arr_point[3])
h2 = hinhchieu(center_bottom[0],arr_point[0], arr_point[3], arr_point[2])
# # print(h1)
cv2.circle(img, (int(h1[0]),int(h1[1])), 4, (255, 200, 0), -1)
cv2.circle(img, (int(h2[0]),int(h2[1])), 4, (255, 200, 0), -1)
x = arr_near[0][0]*2.75 + ti_le(arr_point[1],h1,arr_point[2])+2.75
y = arr_near[0][1]*2.75 + ti_le(h2,center_bottom[0],h1)+2.75

print(x,y)

cv2.imshow("result", img)
cv2.waitKey()