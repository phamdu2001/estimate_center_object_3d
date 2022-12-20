import cv2
import numpy as np
import matplotlib.pyplot as plt

# range color 
def detect_chess_board(path):
    hh = 500
    ww = 400
    image = cv2.imread(path)
        
    # resize image
    image = cv2.resize(image,(hh,ww))

    #1. Tách màu sáng, giữ lại màu đen
    # converting frame(image == BGR) to HSV(hue-saturation-value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 0, 137])
    upper_red = np.array([255, 255, 255])
        
    # Morphological Transformations,Opening and Closing
    thresh = cv2.inRange(hsv,lower_red, upper_red)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(image, image, mask = mask)

    #2. lấy vùng bàn cờ
    contours = cv2.findContours(255 - mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    contours = contours[0] if len(contours) == 2 else contours[1]
    mask_0 = np.zeros((ww,hh), dtype=np.uint8)
    for cntr in contours:
        cv2.drawContours(mask_0, [cntr], 0, (255,255,255), -1)
    result = cv2.bitwise_and(image, image, mask=mask_0)

    #3. Tách trắng lấy đen loại bỏ viền bàn cờ
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([255, 255, 162])
        
    # Morphological Transformations,Opening and Closing
    thresh = cv2.inRange(hsv,lower_red, upper_red)
    kernel = np.ones((15,15),np.uint8)

    #4. khử nhiễu
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    image = np.zeros((ww,hh), dtype=np.uint8)
    image += 255
    result = cv2.bitwise_and(image, image, mask = mask)
    corners = 0
    mask_0 = mask
    #5. Lấy góc ô cờ
    found, corners = cv2.findChessboardCorners(result, (7, 7),flags=cv2.CALIB_CB_FAST_CHECK)
    # print(np.shape(corners))

    # corner_lt = (corners[0]-corners[16])

    img = cv2.imread(path)
    img = cv2.resize(img,(hh,ww))
    img2 = np.copy(img)  # Make a copy of original img as img2

    for corner in corners:
        # print(corner[0])
        coord = (int(corner[0][0]), int(corner[0][1]))
        # print(coord)
        cv2.circle(img=img2, center=coord, radius=2, color=(255, 0, 0), thickness=2)

    corner_lt = np.array([corners[0][0][0]-(corners[8][0][0]-corners[0][0][0])**2/(corners[16][0][0]-corners[8][0][0]),corners[0][0][1]-(corners[8][0][1]-corners[0][0][1])**2/(corners[16][0][1]-corners[8][0][1])])
    corner_rb = np.array([corners[48][0][0]+(corners[48][0][0]-corners[40][0][0])**2/(corners[40][0][0]-corners[32][0][0]),corners[48][0][1]+(corners[48][0][1]-corners[40][0][1])**2/(corners[40][0][1]-corners[32][0][1])])
    corner_lb = np.array([corners[42][0][0]-(corners[36][0][0]-corners[42][0][0])**2/(corners[30][0][0]-corners[36][0][0]),corners[42][0][1]+(corners[42][0][1]-corners[36][0][1])**2/(corners[36][0][1]-corners[30][0][1])])
    corner_rt = np.array([corners[6][0][0]+(corners[6][0][0]-corners[12][0][0])**2/(corners[12][0][0]-corners[18][0][0]),corners[6][0][1]-(corners[12][0][1]-corners[6][0][1])**2/(corners[18][0][1]-corners[12][0][1])])

    mask = np.zeros((ww,hh), dtype=np.uint8)
    mask = cv2.line(mask, (int(corner_lt[0]),int(corner_lt[1])), (int(corner_rt[0]),int(corner_rt[1])), (255,255,255), 2)
    mask = cv2.line(mask, (int(corner_lt[0]),int(corner_lt[1])), (int(corner_lb[0]),int(corner_lb[1])), (255,255,255), 2)
    mask = cv2.line(mask, (int(corner_rb[0]),int(corner_rb[1])), (int(corner_rt[0]),int(corner_rt[1])), (255,255,255), 2)
    mask = cv2.line(mask, (int(corner_rb[0]),int(corner_rb[1])), (int(corner_lb[0]),int(corner_lb[1])), (255,255,255), 2)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    mask_0 = np.zeros((ww,hh), dtype=np.uint8)
    for cntr in contours:
        cv2.drawContours(mask_0, [cntr], 0, (255,255,255), -1)
    
    return mask_0, corners

def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def point_near(point, array):
    near = 10000
    near_index = np.array([0,0])
    for i in range(len(array)):
        dt= distance(point, array[i])
        if near > dt:
            near = dt

            near_index = np.array([i//7,i%7])
    print(near_index)
    arr_near = np.zeros((4,2))
    if point[0] > near_index[0] and point[1]> near_index[1]:
        arr_near[0] = np.array(near_index)
        arr_near[1] = np.array([near_index[0],near_index[1]+1])
        arr_near[2] = np.array([near_index[0]+1,near_index[1]+1])
        arr_near[3] = np.array([near_index[0]+1,near_index[1]])
    elif point[0] < near_index[0] and point[1]> near_index[1]:
        arr_near[0] = np.array([near_index[0],near_index[1]-1])
        arr_near[1] = np.array(near_index)
        arr_near[2] = np.array([near_index[0]+1,near_index[1]])
        arr_near[3] = np.array([near_index[0]+1,near_index[1]-1])
    elif point[0] < near_index[0] and point[1]< near_index[1]:
        arr_near[0] = np.array([near_index[0]-1,near_index[1]-1])
        arr_near[1] = np.array([near_index[0]-1,near_index[1]])
        arr_near[2] = np.array(near_index)
        arr_near[3] = np.array([near_index[0],near_index[1]-1])
    elif point[0] > near_index[0] and point[1]< near_index[1]:
        arr_near[0] = np.array([near_index[0]-1,near_index[1]])
        arr_near[1] = np.array([near_index[0]-1,near_index[1]+1])
        arr_near[2] = np.array([near_index[0],near_index[1]+1])
        arr_near[3] = np.array(near_index)
    print(arr_near)
    arr_point = np.array([array[convert_2d_to_1d(arr_near[0].astype(int))],array[convert_2d_to_1d(arr_near[1].astype(int))],array[convert_2d_to_1d(arr_near[2].astype(int))],array[convert_2d_to_1d(arr_near[3].astype(int))]])
    return near, near_index, arr_near, arr_point

def convert_2d_to_1d(a,n=7):
    return a[0]*n+a[1]

# print(convert_2d_to_1d([3,2]))

def get_line_2point(A,B):
    vector_AB = B - A
    a = 0-vector_AB[1]
    b = vector_AB[0]
    c = -a*A[0] - b*A[1]
    return a,b,c

def check_line(A,B,C):
    a,b,c = get_line_2point(A,B)
    if C[0]*a+C[1]*b+c ==0:
        print("True")
    else:
        print("false")

def check_phuong_vecto(a,b):
    if a[0]/b[0] == a[1]/b[1]:
        print("true")
    else:
        print("false")

def giao_diem(a1,b1,c1,a2,b2,c2):
    y = (c2-(a2*c1/a1))/((a2*b1/a1)-b2)
    x = 0-(c1+b1*y)/a1
    return x,y

# a1,b1,c1=get_line_2point(np.array([0,1]),np.array([2,3]))
# a2,b2,c2=get_line_2point(np.array([0,4]),np.array([2,3]))
# print(giao_diem(a1,b1,c1,a2,b2,c2))

def hinhchieu(C,p1,p2,p3):
    a1,b1,c1 = get_line_2point(p1,p2)
    a2,b2,c2 = get_line_2point(C,p3)

    # print(a2,b2,c2,a1,b1,c1)
    x,y = giao_diem(a1,b1,c1,a2,b2,c2)
    G = np.array([x,y])
    H = C+(p2-p3)*(G-C)/(G-p3)
    return H

def ti_le(a1,a2,a3,n=2.75):
    return distance(a1,a2)/distance(a1,a3)*n
# img2 = cv2.bitwise_and(img2, img2, mask=mask_0)

# path = "D:/Study/Xulyanh/code/data/back.jpg"
# mask, corners = detect_chess_board(path)
# image = cv2.imread(path)
# image = cv2.resize(image,(500,400))
# crop_chess_board = cv2.bitwise_and(image, image, mask=mask)
# for corner in corners:
#         # print(corner[0])
#         coord = (int(corner[0][0]), int(corner[0][1]))
#         # print(coord)
#         cv2.circle(img=crop_chess_board, center=coord, radius=2, color=(255, 0, 0), thickness=2)
# cv2.imshow('mask',crop_chess_board);
# cv2.waitKey(0)

