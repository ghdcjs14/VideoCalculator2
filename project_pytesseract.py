#!/usr/bin/etc python

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import subprocess
from deep_convnet import DeepConvNet

class Recognition:
     def OrganizeImage(self,img_src):

          img=cv2.imread(img_src,cv2.IMREAD_COLOR)

          # convert to gray image
          convert_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          #cv2.imwrite('image_result/gray.jpg', convert_img)

          # 노이즈 제거를 위한 스무딩 작업
          blur = cv2.GaussianBlur(convert_img,(1,1),0)
          #cv2.imwrite('image_result/blur.jpg',blur)

          canny_img=cv2.Canny(blur,0,255)
          #cv2.imwrite('image_result/canny.jpg',canny_img)

          #edge 검출 알고리즘 적용
          cnts, contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
          box1=[]
          box2 = []

          # 이미지를 다듬는 전처리 과정 나누기
          for i in range(len(contours)):
               cnt=contours[i]
               area = cv2.contourArea(cnt)
               x,y,w,h = cv2.boundingRect(cnt)

               rect_area=w*h  #area size
               aspect_ratio = float(w)/h # ratio = width/height
               if(aspect_ratio>=2.0) and (rect_area>=1000):
                    continue;
               elif(x==0) or (y >= 900):
                    continue;
               else:
                   #print(img[y:y+h, x:x+w])
                   box2.append(cv2.resize(convert_img[y:y+h, x:x+w],(28,28)))
                    #print(box2)
                    #print(img[y:y+h, x:x+w])
                    #cv2.imshow('adf', img[y:y+h, x:x+w])
                    #cv2.waitKey(0)
                   box1.append(cv2.boundingRect(cnt))

          for i in range(len(box1)): ##Buble Sort on python
               for j in range(len(box1)-(i+1)):
                    if box1[j][0]>box1[j+1][0]:
                         temp=box1[j]
                         temp2=box2[j]

                         box1[j]=box1[j+1]
                         box2[j]=box2[j+1]

                         box1[j+1]=temp
                         box2[j+1]=temp2

          #to find number measuring length between rectangles
          for m in range(len(box1)):
               count=0
               for n in range(m+1,(len(box1)-1)):
                    delta_x=abs(box1[n+1][0]-box1[m][0])
                    if delta_x > 150:
                         break
                    delta_y =abs(box1[n+1][1]-box1[m][1])
                    if delta_x ==0:
                         delta_x=1
                    if delta_y ==0:
                         delta_y=1
                    gradient =float(delta_y) / float(delta_x)
                    if gradient < 0.25:
                         count+=1

          cv2.imwrite('image_result/snake.jpg',img) #가장 자리 처리 완료

          network = DeepConvNet()
          network.load_params("deep_convnet_params.pkl")
          #print(box1)
          nine = cv2.imread('images/9.jpg', cv2.IMREAD_GRAYSCALE)
          nine = cv2.resize(nine,(28,28))
          #print(nine.shape)
          #cv2.imwrite('image_result/nine.jpg', nine)
          nine = np.array([[nine]])
          #lalal = np.array(box2)
          #print(ndarr.shape)
          #print(nine.shape)


          for box in box2 :
              #print(box)
              #print(box.shape)
              npbox = np.array([[box]])
              y = network.predict(npbox)
              print(np.argmax(y, axis=1))
              cv2.imshow('adf',box)
              cv2.waitKey(0)

          #우리 프로젝트의 경우 적정 길이는 아직 미정
          ret, number_th = cv2.threshold(convert_img,100,255,cv2.THRESH_BINARY)
          cv2.imwrite('image_result/number_th.jpg',number_th)
          kernel = np.ones((3,3))
          er_number = cv2.erode(number_th,kernel,iterations=2)
          cv2.imwrite('image_result/er_number.jpg', er_number)

          return (Image.open('image_result/er_number.jpg'))

     def ExtractNumber(self,img_src):
          #결과 텍스트 파일을 열어 읽습니다.
          txt_result = pytesseract.image_to_string(self.OrganizeImage(img_src))
          f_number = open("image_result2.txt",'w', encoding='UTF-8', newline='')
          f_number.write(txt_result.replace(" ", ""))
          f_number.close()
          return (txt_result)


recogtest=Recognition()
recogtest.OrganizeImage('images/photo_1.jpg')
#result=recogtest.ExtractNumber('images/digits3.jpg')
#print(result)