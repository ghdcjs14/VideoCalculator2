import cv2
import numpy as np

imgGray = cv2.imread('images/sample01.png',cv2.IMREAD_GRAYSCALE)


cv2.imwrite('image_result/sample01.png',cv2.resize(imgGray,(480,680)))

# 2. Morph Gradient - Naver
kernel = np.ones((5,5),np.uint8)
gradient = cv2.morphologyEx(imgGray, cv2.MORPH_GRADIENT, kernel)

image, contours, hierarchy = cv2.findContours(imgGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print(contour)

        cv2.rectangle(imgGray, (x, y), (x + w, y + h), (0, 255, 0), 3)



cv2.imshow('imgGray',imgGray)
cv2.waitKey(0)