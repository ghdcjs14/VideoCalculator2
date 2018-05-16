import cv2
import numpy as np

imgGray = cv2.imread('images/scan-09.jpg',cv2.IMREAD_GRAYSCALE)
cv2.resize(imgGray,(480,680))
cv2.imwrite('images/scan-09.jpg',cv2.resize(imgGray,(480,680)))
imgGray = cv2.imread('images/scan-09.jpg',cv2.IMREAD_GRAYSCALE)

# 2. Erosion
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(imgGray,kernel,iterations=1)
#cv2.imwrite('train_images/02Erosion.jpg', erosion)

# 2. Morph Gradient - Naver
gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel)
#cv2.imwrite('train_images/02Gradient.jpg', gradient)

# 4. Adaptive Threshold : 잡영 제거
thresh = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
#cv2.imwrite('train_images/03Thresh.jpg', thresh)

# 4. Morph Close : 작은 구멍을 메우고 경계를 강화
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#cv2.imwrite('train_images/04Closing.jpg', closing)

# 6. edge 검출 알고리즘 적용
boxes = []
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, 255, 3)
if len(contours) > 0:
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w>25 and h>25 and not (w>43 and h>43) :
            boxes.append(cv2.resize(imgGray[y:y + h, x:x + w], (28, 28)))
            #self.boxes.append(cv2.resize(thresh[y:y + h, x:x + w], (28, 28)))
            cv2.rectangle(imgGray, (x, y), (x + w, y + h), (0, 255, 0), 1)
#cv2.imwrite('train_images/04Contours.jpg', imgGray)

# crop 할 사이즈 : grid_w, grid_h
#grid_w = 68  # crop width
#grid_h = 73  # crop height
#range_w = (int)(480 / grid_w)
#range_h = (int)(680 / grid_h)
#print(range_w, range_h)

i = 0
save_path = "train_images/"

for box in boxes:

#        bbox = (h * grid_h, w * grid_w, (h + 1) * (grid_h), (w + 1) * (grid_w))
#        print(h * grid_h, w * grid_w, (h + 1) * (grid_h), (w + 1) * (grid_w))
#        # 가로 세로 시작, 가로 세로 끝
    #crop_img = imgGray[ box[1]:box[3], box[0]:box[2]]

    fname = "{}.jpg".format("6_{0:05d}".format(i))
    savename = save_path + fname
    cv2.imwrite(savename, box)
    print('save file ' + savename + '....')
    i += 1

#cv2.imshow('imgGray',imgGray)
#cv2.waitKey(0)